use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use std::sync::mpsc;
use std::sync::mpsc::{Sender, Receiver};
use std::thread;
use std::sync::{Arc, RwLock};

use noise::Seed;
use glium::backend::Facade;
use glium::index::PrimitiveType;
use glium::{VertexBuffer, IndexBuffer, Frame, Surface, Program, DrawParameters, BackfaceCullingMode, Depth, DepthTest};
use cgmath::{Matrix4};

use chunk;
use chunk::{Chunk, Vertex, Block};
use position::{ChunkPosition, Position};

pub struct MasterScheduler {
    seed: u32,
    noise_seed: Arc<Seed>,
    
    meshes: HashMap<ChunkPosition, (VertexBuffer<Vertex>, IndexBuffer<u32>)>,
    chunks: HashMap<ChunkPosition, Arc<RwLock<Chunk>>>,
    unmeshed_chunks: HashSet<ChunkPosition>,
    unpopulated_chunks: HashSet<ChunkPosition>,
    
    // Generators
    pub serially_generate: bool,
    serial_generator: SerialGenerator,
    parallel_generator: ParallelGenerator,
    
    // Populators
    pub serially_populate: bool,
    serial_populator: SerialPopulator,
    parallel_populator: ParallelPopulator,
    
    // Mesh Constructors
    pub serially_construct: bool,
    serial_constructor: SerialMeshConstructor,
    parallel_constructor: ParallelMeshConstructor
}

impl MasterScheduler {
    pub fn new() -> MasterScheduler {
        let seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as u32;
        MasterScheduler {
            seed: seed,
            noise_seed: Arc::new(Seed::new(seed)),
            meshes: HashMap::new(),
            chunks: HashMap::new(),
            unmeshed_chunks: HashSet::new(),
            unpopulated_chunks: HashSet::new(),
            serially_generate: false,
            serial_generator: SerialGenerator,
            parallel_generator: ParallelGenerator::new(),
            serially_populate: false,
            serial_populator: SerialPopulator,
            parallel_populator: ParallelPopulator::new(),
            serially_construct: false,
            serial_constructor: SerialMeshConstructor,
            parallel_constructor: ParallelMeshConstructor::new()
        }
    }
    
    pub fn update<F: Facade>(&mut self, camera_pos: [f32; 3], facade: &F) {
        let radius = 10;
        
        let camera_chunk_pos = [(camera_pos[0] / 16.0).floor() as i32, (camera_pos[1] / 16.0).floor() as i32];
        
        // Trim chunks outside the range of the camera
        
        let mut removed = Vec::new();
        
        for &pos in self.chunks.keys() {
            if pos.x < camera_chunk_pos[0] - radius ||
            pos.x > camera_chunk_pos[0] + radius + 1 ||
            pos.y < camera_chunk_pos[1] - radius ||
            pos.y > camera_chunk_pos[1] + radius + 1 {
                removed.push(pos);
            }
        }
        
        if removed.len() != 0 {
            println!("Removed: {}", removed.len());
        }
        
        for pos in removed {
            self.chunks.remove(&pos);
            self.meshes.remove(&pos);
            self.unmeshed_chunks.remove(&pos);
            self.unpopulated_chunks.remove(&pos);
        }
        
        // Generate chunks in range of the camera
        
        let mut generating = Vec::new();
        
        'outer: for x in camera_chunk_pos[0] - radius .. camera_chunk_pos[0] + radius + 1 {
            for y in camera_chunk_pos[1] - radius .. camera_chunk_pos[1] + radius + 1 {
                let pos = ChunkPosition {x: x, y: y};
                if !self.chunks.contains_key(&pos) {
                    generating.push(pos);
                    //break 'outer;
                }
            }
        }
        
        if generating.len() != 0 {
            let now = Instant::now();
            
            if self.serially_generate {
                self.serial_generator.generate(&generating, &mut self.chunks, self.noise_seed.clone());
            } else {
                self.parallel_generator.generate(&generating, &mut self.chunks, self.noise_seed.clone());
            }
            
            let duration = Instant::now() - now;
            
            println!("Generated: {} in {:.2} ms", generating.len(), duration.as_secs() as f64 * 1000.0 + duration.subsec_nanos() as f64 / 1000000.0);
        }
        
        self.unpopulated_chunks.extend(generating);
        
        // Populate chunks that have all surrounding chunks generated
        
        let mut populating = Vec::new();
        
        for &pos in &self.unpopulated_chunks {
            if self.chunks.contains_key(&ChunkPosition {x: pos.x + 1, y: pos.y + 1}) &&
            self.chunks.contains_key(&ChunkPosition {x: pos.x + 0, y: pos.y + 1}) &&
            self.chunks.contains_key(&ChunkPosition {x: pos.x - 1, y: pos.y + 1}) &&
            self.chunks.contains_key(&ChunkPosition {x: pos.x + 1, y: pos.y + 0}) &&
            self.chunks.contains_key(&ChunkPosition {x: pos.x - 1, y: pos.y + 0}) &&
            self.chunks.contains_key(&ChunkPosition {x: pos.x + 1, y: pos.y - 1}) &&
            self.chunks.contains_key(&ChunkPosition {x: pos.x + 0, y: pos.y - 1}) &&
            self.chunks.contains_key(&ChunkPosition {x: pos.x - 1, y: pos.y - 1}) {
                populating.push(pos);
                //break;
            }
        }
        
        for pos in &populating {
            self.unpopulated_chunks.remove(pos);
        }
        
        if populating.len() != 0 {
            let now = Instant::now();
            
            if self.serially_populate {
                self.serial_populator.populate(&populating, &self.chunks, self.seed);
            } else {
                self.parallel_populator.populate(&populating, &self.chunks, self.seed);
            }
            
            let duration = Instant::now() - now;
            
            println!("Populated: {} in {:.2} ms", populating.len(), duration.as_secs() as f64 * 1000.0 + duration.subsec_nanos() as f64 / 1000000.0);
        }
        
        self.unmeshed_chunks.extend(populating);
        
        // Construct meshes for all chunks with all their neighbors populated
        
        let mut meshing = Vec::new();
        
        for &pos in &self.unmeshed_chunks {
            if !self.unpopulated_chunks.contains(&ChunkPosition {x: pos.x + 1, y: pos.y + 1}) &&
            self.chunks.contains_key(&ChunkPosition {x: pos.x + 1, y: pos.y + 1}) &&
            !self.unpopulated_chunks.contains(&ChunkPosition {x: pos.x + 0, y: pos.y + 1}) &&
            self.chunks.contains_key(&ChunkPosition {x: pos.x + 0, y: pos.y + 1}) &&
            !self.unpopulated_chunks.contains(&ChunkPosition {x: pos.x - 1, y: pos.y + 1}) &&
            self.chunks.contains_key(&ChunkPosition {x: pos.x - 1, y: pos.y + 1}) &&
            !self.unpopulated_chunks.contains(&ChunkPosition {x: pos.x + 1, y: pos.y + 0}) &&
            self.chunks.contains_key(&ChunkPosition {x: pos.x + 1, y: pos.y + 0}) &&
            !self.unpopulated_chunks.contains(&ChunkPosition {x: pos.x - 1, y: pos.y + 0}) &&
            self.chunks.contains_key(&ChunkPosition {x: pos.x - 1, y: pos.y + 0}) &&
            !self.unpopulated_chunks.contains(&ChunkPosition {x: pos.x + 1, y: pos.y - 1}) &&
            self.chunks.contains_key(&ChunkPosition {x: pos.x + 1, y: pos.y - 1}) &&
            !self.unpopulated_chunks.contains(&ChunkPosition {x: pos.x + 0, y: pos.y - 1}) &&
            self.chunks.contains_key(&ChunkPosition {x: pos.x + 0, y: pos.y - 1}) &&
            !self.unpopulated_chunks.contains(&ChunkPosition {x: pos.x - 1, y: pos.y - 1}) &&
            self.chunks.contains_key(&ChunkPosition {x: pos.x - 1, y: pos.y - 1}) {
                meshing.push(pos);
            }
        }
        
        for pos in &meshing {
            self.unmeshed_chunks.remove(pos);
        }
        
        if meshing.len() != 0 {
            let now = Instant::now();
            
            if self.serially_construct {
                self.serial_constructor.construct_meshes(&meshing, &self.chunks, &mut self.meshes, facade);
            } else {
                self.parallel_constructor.construct_meshes(&meshing, &self.chunks, &mut self.meshes, facade);
            }
            
            let duration = Instant::now() - now;
            
            println!("Meshed: {} in {:.2} ms", meshing.len(), duration.as_secs() as f64 * 1000.0 + duration.subsec_nanos() as f64 / 1000000.0);
        }
    }
    
    pub fn render(&mut self, frame: &mut Frame, program: &Program, model_view_projection: Matrix4<f32>) {
        let params = DrawParameters {
            depth: Depth {
               test: DepthTest::IfLess,
               write: true,
               .. Default::default()
           },
            backface_culling: BackfaceCullingMode::CullingDisabled,
            .. Default::default()
        };
        
        let model_view_projection: [[f32; 4]; 4] = model_view_projection.into();
        
        for (pos, &(ref vertex_buffer, ref index_buffer)) in self.meshes.iter() {
            let uniforms = uniform! {
                model_view_projection: model_view_projection,
                chunk_position: [pos.x as f32, pos.y as f32]
            };
            frame.draw(vertex_buffer, index_buffer, program, &uniforms, &params).unwrap();
        }
    }
    
    pub fn get_height(&self, xy_pos: [i32; 2]) -> usize {
        let chunk_pos = ChunkPosition {x: xy_pos[0] >> 4, y: xy_pos[1] >> 4};
        
        if let Some(chunk) = self.chunks.get(&chunk_pos) {
            let chunk = chunk.read().unwrap();
            
            let mut height = 127;
            
            while height > 0 {
                if chunk.0[xy_pos[0] as usize & 0xF][xy_pos[1] as usize & 0xF][height] != Block::Air {
                    break;
                }
                
                height -= 1;
            }
            
            height
        } else {
            128
        }
    }
}

trait Generator {
    fn generate(&mut self, generating: &[ChunkPosition], chunks: &mut HashMap<ChunkPosition, Arc<RwLock<Chunk>>>, seed: Arc<Seed>);
}

struct SerialGenerator;

impl Generator for SerialGenerator {
    fn generate(&mut self, generating: &[ChunkPosition], chunks: &mut HashMap<ChunkPosition, Arc<RwLock<Chunk>>>, seed: Arc<Seed>) {
        for &pos in generating {
            chunks.insert(pos, chunk::generate(pos, &seed));
        }
    }
}

struct ParallelGenerator {
    worker_senders: Vec<Sender<(ChunkPosition, Arc<Seed>)>>,
    chunk_receiver: Receiver<(ChunkPosition, Arc<RwLock<Chunk>>)>
}

impl ParallelGenerator {
    fn new() -> ParallelGenerator {
        let thread_count = 4;
        
        let (chunk_sender, chunk_receiver) = mpsc::channel();
        let mut worker_senders = Vec::new();
        
        for _ in 0..thread_count {
            let (worker_sender, worker_receiver) = mpsc::channel::<(ChunkPosition, Arc<Seed>)>();
            worker_senders.push(worker_sender);
            let chunk_sender = chunk_sender.clone();
            
            thread::spawn(move || {
                for (pos, seed) in &worker_receiver {
                    chunk_sender.send((pos, chunk::generate(pos, &seed))).unwrap();
                }
            });
        }
        
        ParallelGenerator {worker_senders: worker_senders, chunk_receiver: chunk_receiver}
    }
}

impl Generator for ParallelGenerator {
    fn generate(&mut self, generating: &[ChunkPosition], chunks: &mut HashMap<ChunkPosition, Arc<RwLock<Chunk>>>, seed: Arc<Seed>) {
        let thread_count = self.worker_senders.len();
        
        for (i, &pos) in generating.iter().enumerate() {
            self.worker_senders[i % thread_count].send((pos, seed.clone())).unwrap();
        }
        
        let mut generating_set = HashSet::<ChunkPosition>::new();
        generating_set.extend(generating);
        
        while generating_set.len() > 0 {
            let (pos, chunk) = self.chunk_receiver.recv().unwrap();
            chunks.insert(pos, chunk);
            generating_set.remove(&pos);
        }
    }
}

trait Populator {
    fn populate(&mut self, populating: &[ChunkPosition], chunks: &HashMap<ChunkPosition, Arc<RwLock<Chunk>>>, seed: u32);
}

struct SerialPopulator;

impl Populator for SerialPopulator {
    fn populate(&mut self, populating: &[ChunkPosition], chunks: &HashMap<ChunkPosition, Arc<RwLock<Chunk>>>, seed: u32) {
        for &pos in populating {
            let mut changes = HashMap::new();
            chunk::populate(chunks.get(&pos).unwrap().clone(), pos, seed, |pos, block_pos, block| {
                changes.entry(pos).or_insert(Vec::new()).push((block_pos, block));
            });
            
            for (pos, changes) in changes {
                let mut chunk = chunks.get(&pos).unwrap().write().unwrap();
                for (block_pos, block) in changes {
                    if chunk.0[block_pos.x as usize][block_pos.y as usize][block_pos.z as usize] == Block::Air {
                        chunk.0[block_pos.x as usize][block_pos.y as usize][block_pos.z as usize] = block;
                    }
                }
            }
        }
    }
}

enum ChunkPopulationRequest {
    Change(Position, Block),
    Done
}

struct ParallelPopulator {
    worker_senders: Vec<Sender<(ChunkPosition, u32, Arc<RwLock<Chunk>>)>>,
    chunk_receiver: Receiver<(ChunkPosition, ChunkPopulationRequest)>
}

impl ParallelPopulator {
    fn new() -> ParallelPopulator {
        let thread_count = 4;
        
        let (chunk_sender, chunk_receiver) = mpsc::channel();
        let mut worker_senders = Vec::new();
        
        for _ in 0..thread_count {
            let (worker_sender, worker_receiver) = mpsc::channel();
            worker_senders.push(worker_sender);
            let chunk_sender = chunk_sender.clone();
            
            thread::spawn(move || {
                for (pos, seed, chunk) in &worker_receiver {
                    chunk::populate(chunk, pos, seed, |pos, block_pos, block| {
                        chunk_sender.send((pos, ChunkPopulationRequest::Change(block_pos, block))).unwrap();
                    });
                    
                    chunk_sender.send((pos, ChunkPopulationRequest::Done)).unwrap();
                }
            });
        }
        
        ParallelPopulator {worker_senders: worker_senders, chunk_receiver: chunk_receiver}
    }
}

impl Populator for ParallelPopulator {
    fn populate(&mut self, populating: &[ChunkPosition], chunks: &HashMap<ChunkPosition, Arc<RwLock<Chunk>>>, seed: u32) {
        let thread_count = self.worker_senders.len();
        
        for (i, &pos) in populating.iter().enumerate() {
            self.worker_senders[i % thread_count].send((pos, seed, chunks.get(&pos).unwrap().clone())).unwrap();
        }
        
        let mut changes = HashMap::new();
        
        let mut populating_set = HashSet::<ChunkPosition>::new();
        populating_set.extend(populating);
        
        while populating_set.len() > 0 {
            let (pos, command) = self.chunk_receiver.recv().unwrap();
            match command {
                ChunkPopulationRequest::Change(block_pos, block) => {
                    if let Ok(mut chunk) = chunks.get(&pos).unwrap().try_write() {
                        if chunk.0[block_pos.x as usize][block_pos.y as usize][block_pos.z as usize] == Block::Air {
                            chunk.0[block_pos.x as usize][block_pos.y as usize][block_pos.z as usize] = block;
                        }
                    } else {
                        changes.entry(pos).or_insert(Vec::new()).push((block_pos, block));
                    }
                },
                ChunkPopulationRequest::Done => {
                    populating_set.remove(&pos);
                }
            }
        }
        
        for (pos, changes) in changes {
            let mut chunk = chunks.get(&pos).unwrap().write().unwrap();
            for (block_pos, block) in changes {
                if chunk.0[block_pos.x as usize][block_pos.y as usize][block_pos.z as usize] == Block::Air {
                    chunk.0[block_pos.x as usize][block_pos.y as usize][block_pos.z as usize] = block;
                }
            }
        }
    }
}

trait MeshConstructor {
    fn construct_meshes<F: Facade>(&mut self, constructing: &[ChunkPosition], chunks: &HashMap<ChunkPosition, Arc<RwLock<Chunk>>>, meshes: &mut HashMap<ChunkPosition, (VertexBuffer<Vertex>, IndexBuffer<u32>)>, facade: &F);
}

struct SerialMeshConstructor;

impl MeshConstructor for SerialMeshConstructor {
    fn construct_meshes<F: Facade>(&mut self, constructing: &[ChunkPosition], chunks: &HashMap<ChunkPosition, Arc<RwLock<Chunk>>>, meshes: &mut HashMap<ChunkPosition, (VertexBuffer<Vertex>, IndexBuffer<u32>)>, facade: &F) {
        for &pos in constructing {
            let (vertices, indices) = chunk::construct_mesh(chunks.get(&pos).unwrap().clone(), pos, |pos, block_pos| {
                chunks.get(&pos).unwrap().read().unwrap().0[block_pos.x as usize][block_pos.y as usize][block_pos.z as usize]
            });
            
            meshes.insert(pos, (VertexBuffer::new(facade, &vertices).unwrap(), IndexBuffer::new(facade, PrimitiveType::TrianglesList, &indices).unwrap()));
        }
    }
}

struct ParallelMeshConstructor {
    worker_senders: Vec<Sender<(ChunkPosition, Arc<RwLock<Chunk>>, [Arc<RwLock<Chunk>>; 8])>>,
    chunk_receiver: Receiver<(ChunkPosition, Vec<Vertex>, Vec<u32>)>
}

impl ParallelMeshConstructor {
    fn new() -> ParallelMeshConstructor {
        let thread_count = 4;
        
        let (chunk_sender, chunk_receiver) = mpsc::channel();
        let mut worker_senders = Vec::new();
        
        for _ in 0..thread_count {
            let (worker_sender, worker_receiver) = mpsc::channel::<(ChunkPosition, Arc<RwLock<Chunk>>, [Arc<RwLock<Chunk>>; 8])>();
            worker_senders.push(worker_sender);
            let chunk_sender = chunk_sender.clone();
            
            thread::spawn(move || {
                for (pos, chunk, surrounding) in &worker_receiver {
                    let (vertices, indices) = chunk::construct_mesh(chunk, pos, |new_pos, block_pos| {
                        match (new_pos.x - pos.x, new_pos.y - pos.y) {
                            (0, 1) => {
                                surrounding[0].read().unwrap().0[block_pos.x as usize][block_pos.y as usize][block_pos.z as usize]
                            },
                            (1, 1) => {
                                surrounding[1].read().unwrap().0[block_pos.x as usize][block_pos.y as usize][block_pos.z as usize]
                            },
                            (1, 0) => {
                                surrounding[2].read().unwrap().0[block_pos.x as usize][block_pos.y as usize][block_pos.z as usize]
                            },
                            (1, -1) => {
                                surrounding[3].read().unwrap().0[block_pos.x as usize][block_pos.y as usize][block_pos.z as usize]
                            },
                            (0, -1) => {
                                surrounding[4].read().unwrap().0[block_pos.x as usize][block_pos.y as usize][block_pos.z as usize]
                            },
                            (-1, -1) => {
                                surrounding[5].read().unwrap().0[block_pos.x as usize][block_pos.y as usize][block_pos.z as usize]
                            },
                            (-1, 0) => {
                                surrounding[6].read().unwrap().0[block_pos.x as usize][block_pos.y as usize][block_pos.z as usize]
                            },
                            (-1, 1) => {
                                surrounding[7].read().unwrap().0[block_pos.x as usize][block_pos.y as usize][block_pos.z as usize]
                            },
                            x => {
                                unreachable!("Read invalid chunk displacement: {:?}", x)
                            }
                        }
                    });
                    chunk_sender.send((pos, vertices, indices)).unwrap();
                }
            });
        }
        
        ParallelMeshConstructor {worker_senders: worker_senders, chunk_receiver: chunk_receiver}
    }
}

impl MeshConstructor for ParallelMeshConstructor {
    fn construct_meshes<F: Facade>(&mut self, constructing: &[ChunkPosition], chunks: &HashMap<ChunkPosition, Arc<RwLock<Chunk>>>, meshes: &mut HashMap<ChunkPosition, (VertexBuffer<Vertex>, IndexBuffer<u32>)>, facade: &F) {
        let thread_count = self.worker_senders.len();
        
        for (i, &pos) in constructing.iter().enumerate() {
            self.worker_senders[i % thread_count].send((pos, chunks.get(&pos).unwrap().clone(),
            [chunks.get(&pos.north()).unwrap().clone(), chunks.get(&pos.north_east()).unwrap().clone(),
            chunks.get(&pos.east()).unwrap().clone(), chunks.get(&pos.south_east()).unwrap().clone(),
            chunks.get(&pos.south()).unwrap().clone(), chunks.get(&pos.south_west()).unwrap().clone(),
            chunks.get(&pos.west()).unwrap().clone(), chunks.get(&pos.north_west()).unwrap().clone()])).unwrap();
        }
        
        let mut constructing_set = HashSet::<ChunkPosition>::new();
        constructing_set.extend(constructing);
        
        while constructing_set.len() > 0 {
            let (pos, vertices, indices) = self.chunk_receiver.recv().expect("1");
            
            meshes.insert(pos, (VertexBuffer::new(facade, &vertices).expect("2"), IndexBuffer::new(facade, PrimitiveType::TrianglesList, &indices).expect("3")));
            
            constructing_set.remove(&pos);
        }
    }
}
