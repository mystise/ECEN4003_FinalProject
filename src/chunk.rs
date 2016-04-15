use std::sync::{Arc, RwLock};

use noise::{open_simplex2, Seed, Brownian2};
use rand::{XorShiftRng, SeedableRng, Rng};
use rand::distributions::{Range, IndependentSample};

use position::{ChunkPosition, Position};

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum Block {
    Air,
    Stone,
    Dirt,
    Grass,
    Log,
    Leaf,
    Water
}

impl Block {
    fn color(&self) -> [u8; 4] {
        match *self {
            Block::Air => {
                [0x00, 0x00, 0x00, 0x00]
            },
            Block::Stone => {
                // #708090 slate gray
                [0x70, 0x80, 0x90, 0xFF]
            },
            Block::Dirt => {
                // #80461B russet brown
                [0x80, 0x46, 0x1B, 0xFF]
            },
            Block::Grass => {
                // #0a8504
                [0x0A, 0x85, 0x04, 0xFF]
            },
            Block::Log => {
                // #4d3312
                [0x4D, 0x33, 0x12, 0xFF]
            },
            Block::Leaf => {
                // #175c17
                [0x17, 0x5C, 0x17, 0xFF]
            },
            Block::Water => {
                // #0057ff
                [0x00, 0x57, 0xFF, 0xFF]
            }
        }
    }
}

#[derive(Clone)]
pub struct Chunk(pub [[[Block; 128]; 16]; 16]);

impl Chunk {
    pub fn new() -> Chunk {
        Chunk([[[Block::Air; 128]; 16]; 16])
    }
}

pub fn generate(pos: ChunkPosition, seed: &Seed) -> Arc<RwLock<Chunk>> {
    let mut output = Chunk::new();
    
    let height_noise = Brownian2::new(open_simplex2, 4).frequency(1.0/150.0);
    
    for x in 0..16i32 {
        for y in 0..16i32 {
            let xy_pos = (pos.x * 16 + x, pos.y * 16 + y);
            
            let height = {
                let noise = height_noise.apply(seed, &[xy_pos.0 as f32, xy_pos.1 as f32]);
                ((noise + 1.0) / 2.0 * 32.0 + 16.0) as i32
            };
            
            for z in 0..height {
                output.0[x as usize][y as usize][z as usize] = Block::Stone;
            }
            
            let dirt_height = {
                let scale = 32.0;
                let noise = open_simplex2(seed, &[xy_pos.0 as f32 / scale, xy_pos.1 as f32 / scale]);
                ((noise + 1.0) / 2.0 * 8.0 - 1.0) as i32
            };
            
            for z in height - dirt_height..height {
                output.0[x as usize][y as usize][z as usize] = Block::Dirt;
            }
            
            if dirt_height > 0 {
                output.0[x as usize][y as usize][height as usize] = Block::Grass;
            }
            
            for z in height..32 {
                output.0[x as usize][y as usize][z as usize] = Block::Water;
            }
        }
    }
    
    Arc::new(RwLock::new(output))
}

pub fn populate<F>(chunk: Arc<RwLock<Chunk>>, pos: ChunkPosition, seed: u32, mut set_block: F) where F: FnMut(ChunkPosition, Position, Block) -> () {
    let mut chunk = chunk.write().unwrap();
    
    let mut rand = XorShiftRng::from_seed([(pos.x as u32).wrapping_mul(0x0CFF235B).wrapping_add((pos.y as u32).wrapping_mul(0x4F72AC03)).wrapping_add(seed), (pos.y as u32).wrapping_mul(0x0CFF235B), (pos.x as u32).wrapping_mul(0x4F72AC03), seed]);
    let pos_range = Range::new(0, 16);
    let height_range = Range::new(5, 10);
    
    for _ in 0..rand.gen_range(0, 12) {
        let block_pos = Position {x: pos_range.ind_sample(&mut rand), y: pos_range.ind_sample(&mut rand), z: 0};
        
        let zpos = {
            let mut zpos = 127;
            while zpos > 0 {
                if chunk.0[block_pos.x as usize][block_pos.y as usize][zpos] != Block::Air {
                    break;
                }
                zpos -= 1;
            }
            zpos
        };
        
        if chunk.0[block_pos.x as usize][block_pos.y as usize][zpos] != Block::Grass {
            continue;
        }
        
        let tree_height = height_range.ind_sample(&mut rand) as usize;
        
        for z in zpos + 1..zpos + tree_height + 1 {
            if chunk.0[block_pos.x as usize][block_pos.y as usize][z] == Block::Air {
                chunk.0[block_pos.x as usize][block_pos.y as usize][z] = Block::Log;
            }
        }
        
        let mut check_set_leaf = |mut pos: ChunkPosition, mut block_pos: Position| {
            let mut changed = false;
            
            if block_pos.x >= 16 {
                block_pos.x -= 16;
                pos.x += 1;
                changed = true;
            }
            
            if block_pos.x < 0 {
                block_pos.x += 16;
                pos.x -= 1;
                changed = true;
            }
            
            if block_pos.y >= 16 {
                block_pos.y -= 16;
                pos.y += 1;
                changed = true;
            }
            
            if block_pos.y < 0 {
                block_pos.y += 16;
                pos.y -= 1;
                changed = true;
            }
            
            if changed {
                set_block(pos, block_pos, Block::Leaf);
            } else {
                if chunk.0[block_pos.x as usize][block_pos.y as usize][block_pos.z as usize] == Block::Air {
                    chunk.0[block_pos.x as usize][block_pos.y as usize][block_pos.z as usize] = Block::Leaf;
                }
            }
        };
        
        for z in zpos + tree_height - 1..zpos + tree_height + 1 {
            for w in -2..3 {
                check_set_leaf(pos, Position {x: block_pos.x + w, y: block_pos.y, z: z as i32});
                check_set_leaf(pos, Position {x: block_pos.x + w, y: block_pos.y + 1, z: z as i32});
                check_set_leaf(pos, Position {x: block_pos.x + w, y: block_pos.y - 1, z: z as i32});
                if w != -2 && w != 2 {
                    check_set_leaf(pos, Position {x: block_pos.x + w, y: block_pos.y + 2, z: z as i32});
                    check_set_leaf(pos, Position {x: block_pos.x + w, y: block_pos.y - 2, z: z as i32});
                }
            }
        }
        
        for w in -1..2 {
            check_set_leaf(pos, Position {x: block_pos.x + w, y: block_pos.y, z: (zpos + tree_height + 1) as i32});
            if w != -1 && w != 1 {
                check_set_leaf(pos, Position {x: block_pos.x + w, y: block_pos.y + 1, z: (zpos + tree_height + 1) as i32});
                check_set_leaf(pos, Position {x: block_pos.x + w, y: block_pos.y - 1, z: (zpos + tree_height + 1) as i32});
            }
        }
    }
}

pub fn construct_mesh<F>(chunk: Arc<RwLock<Chunk>>, pos: ChunkPosition, mut get_block: F) -> (Vec<Vertex>, Vec<u32>) where F: FnMut(ChunkPosition, Position) -> Block {
    let chunk = chunk.read().unwrap();
    
    let mut block_exists = |mut pos: ChunkPosition, mut block_pos: Position| {
        if block_pos.z < 0 || block_pos.z >= 128 {
            return false;
        }
        
        let mut changed = false;
        
        if block_pos.x >= 16 {
            block_pos.x -= 16;
            pos.x += 1;
            changed = true;
        }
        
        if block_pos.x < 0 {
            block_pos.x += 16;
            pos.x -= 1;
            changed = true;
        }
        
        if block_pos.y >= 16 {
            block_pos.y -= 16;
            pos.y += 1;
            changed = true;
        }
        
        if block_pos.y < 0 {
            block_pos.y += 16;
            pos.y -= 1;
            changed = true;
        }
        
        if changed {
            get_block(pos, block_pos) != Block::Air
        } else {
            chunk.0[block_pos.x as usize][block_pos.y as usize][block_pos.z as usize] != Block::Air
        }
    };
    
    let get_face = |base_index: u32, flipped: bool| {
        match flipped {
            false => {
                //3 - 2
                //| ╲ |
                //0 - 1
                [base_index + 3, base_index + 0, base_index + 1, base_index + 1, base_index + 2, base_index + 3]
            },
            true => {
                //3 - 2
                //| ╱ |
                //0 - 1
                [base_index + 0, base_index + 1, base_index + 2, base_index + 2, base_index + 3, base_index + 0]
            }
        }
    };
    
    let get_occlusion = |exists1: bool, exists2: bool, exists3: bool| {
        match (exists1, exists2, exists3) {
            (false, false, false) => {
                3i8
            },
            (true, false, false) | (false, true, false) | (false, false, true) => {
                2
            },
            (true, true, false) | (false, true, true) => {
                1
            },
            (true, _, true) => {
                0
            }
        }
    };
    
    let get_vertex = |x: usize, y: usize, z: usize, color: [u8; 4], occlusion: i8| {
        let occlusion = match occlusion {
            3 => {
                1.0f32
            },
            2 => {
                0.9
            },
            1 => {
                0.8
            },
            0 => {
                0.7
            },
            _ => {
                unreachable!("Invalid occlusion type")
            }
        };
        
        Vertex {position: [x as u8, y as u8, z as u8], color: color, occlusion: occlusion}
    };
    
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    
    for x in 0..16 {
        for y in 0..16 {
            for z in 0..128 {
                let block = chunk.0[x][y][z];
                if block == Block::Air {
                    continue;
                }
                
                let surrounding = [
                block_exists(pos, Position {x: x as i32 - 1, y: y as i32 + 1, z: z as i32 + 1}), // 0
                block_exists(pos, Position {x: x as i32 + 0, y: y as i32 + 1, z: z as i32 + 1}),
                block_exists(pos, Position {x: x as i32 + 1, y: y as i32 + 1, z: z as i32 + 1}),
                
                block_exists(pos, Position {x: x as i32 - 1, y: y as i32 + 0, z: z as i32 + 1}), // 3
                block_exists(pos, Position {x: x as i32 + 0, y: y as i32 + 0, z: z as i32 + 1}),
                block_exists(pos, Position {x: x as i32 + 1, y: y as i32 + 0, z: z as i32 + 1}),
                
                block_exists(pos, Position {x: x as i32 - 1, y: y as i32 - 1, z: z as i32 + 1}), // 6
                block_exists(pos, Position {x: x as i32 + 0, y: y as i32 - 1, z: z as i32 + 1}),
                block_exists(pos, Position {x: x as i32 + 1, y: y as i32 - 1, z: z as i32 + 1}),
                
                block_exists(pos, Position {x: x as i32 - 1, y: y as i32 + 1, z: z as i32 + 0}), // 9
                block_exists(pos, Position {x: x as i32 + 0, y: y as i32 + 1, z: z as i32 + 0}),
                block_exists(pos, Position {x: x as i32 + 1, y: y as i32 + 1, z: z as i32 + 0}),
                
                block_exists(pos, Position {x: x as i32 - 1, y: y as i32 + 0, z: z as i32 + 0}), // 12
                true,
                block_exists(pos, Position {x: x as i32 + 1, y: y as i32 + 0, z: z as i32 + 0}),
                
                block_exists(pos, Position {x: x as i32 - 1, y: y as i32 - 1, z: z as i32 + 0}), // 15
                block_exists(pos, Position {x: x as i32 + 0, y: y as i32 - 1, z: z as i32 + 0}),
                block_exists(pos, Position {x: x as i32 + 1, y: y as i32 - 1, z: z as i32 + 0}),
                
                block_exists(pos, Position {x: x as i32 - 1, y: y as i32 + 1, z: z as i32 - 1}), // 18
                block_exists(pos, Position {x: x as i32 + 0, y: y as i32 + 1, z: z as i32 - 1}),
                block_exists(pos, Position {x: x as i32 + 1, y: y as i32 + 1, z: z as i32 - 1}),
                
                block_exists(pos, Position {x: x as i32 - 1, y: y as i32 + 0, z: z as i32 - 1}), // 21
                block_exists(pos, Position {x: x as i32 + 0, y: y as i32 + 0, z: z as i32 - 1}),
                block_exists(pos, Position {x: x as i32 + 1, y: y as i32 + 0, z: z as i32 - 1}),
                
                block_exists(pos, Position {x: x as i32 - 1, y: y as i32 - 1, z: z as i32 - 1}), // 24
                block_exists(pos, Position {x: x as i32 + 0, y: y as i32 - 1, z: z as i32 - 1}),
                block_exists(pos, Position {x: x as i32 + 1, y: y as i32 - 1, z: z as i32 - 1}),
                ];
                
                if !surrounding[4] {
                    // Top face
                    
                    //3 - 2
                    //| ╲ |
                    //0 - 1
                    
                    let occ0 = get_occlusion(surrounding[3], surrounding[6], surrounding[7]);
                    let occ1 = get_occlusion(surrounding[7], surrounding[8], surrounding[5]);
                    let occ2 = get_occlusion(surrounding[5], surrounding[2], surrounding[1]);
                    let occ3 = get_occlusion(surrounding[1], surrounding[0], surrounding[3]);
                    
                    indices.extend(&get_face(vertices.len() as u32, (occ0 - occ2).abs() < (occ1 - occ3).abs()));
                    
                    vertices.push(get_vertex(x + 0, y + 0, z + 1, block.color(), occ0));
                    vertices.push(get_vertex(x + 1, y + 0, z + 1, block.color(), occ1));
                    vertices.push(get_vertex(x + 1, y + 1, z + 1, block.color(), occ2));
                    vertices.push(get_vertex(x + 0, y + 1, z + 1, block.color(), occ3));
                }
                
                if !surrounding[22] {
                    // Bottom face
                    
                    //3 - 2
                    //| ╲ |
                    //0 - 1
                    
                    let occ0 = get_occlusion(surrounding[21], surrounding[18], surrounding[19]);
                    let occ1 = get_occlusion(surrounding[19], surrounding[20], surrounding[23]);
                    let occ2 = get_occlusion(surrounding[23], surrounding[26], surrounding[25]);
                    let occ3 = get_occlusion(surrounding[25], surrounding[24], surrounding[21]);
                    
                    indices.extend(&get_face(vertices.len() as u32, (occ0 - occ2).abs() < (occ1 - occ3).abs()));
                    
                    vertices.push(get_vertex(x + 0, y + 1, z + 0, block.color(), occ0));
                    vertices.push(get_vertex(x + 1, y + 1, z + 0, block.color(), occ1));
                    vertices.push(get_vertex(x + 1, y + 0, z + 0, block.color(), occ2));
                    vertices.push(get_vertex(x + 0, y + 0, z + 0, block.color(), occ3));
                }
                
                if !surrounding[10] {
                    // North face
                    
                    //3 - 2
                    //| ╲ |
                    //0 - 1
                    
                    let occ0 = get_occlusion(surrounding[11], surrounding[20], surrounding[19]);
                    let occ1 = get_occlusion(surrounding[19], surrounding[18], surrounding[9]);
                    let occ2 = get_occlusion(surrounding[9], surrounding[0], surrounding[1]);
                    let occ3 = get_occlusion(surrounding[1], surrounding[2], surrounding[11]);
                    
                    indices.extend(&get_face(vertices.len() as u32, (occ0 - occ2).abs() < (occ1 - occ3).abs()));
                    
                    vertices.push(get_vertex(x + 1, y + 1, z + 0, block.color(), occ0));
                    vertices.push(get_vertex(x + 0, y + 1, z + 0, block.color(), occ1));
                    vertices.push(get_vertex(x + 0, y + 1, z + 1, block.color(), occ2));
                    vertices.push(get_vertex(x + 1, y + 1, z + 1, block.color(), occ3));
                }
                
                if !surrounding[14] {
                    // East face
                    
                    //3 - 2
                    //| ╲ |
                    //0 - 1
                    
                    let occ0 = get_occlusion(surrounding[17], surrounding[26], surrounding[23]);
                    let occ1 = get_occlusion(surrounding[23], surrounding[20], surrounding[11]);
                    let occ2 = get_occlusion(surrounding[11], surrounding[2], surrounding[5]);
                    let occ3 = get_occlusion(surrounding[5], surrounding[8], surrounding[17]);
                    
                    indices.extend(&get_face(vertices.len() as u32, (occ0 - occ2).abs() < (occ1 - occ3).abs()));
                    
                    vertices.push(get_vertex(x + 1, y + 0, z + 0, block.color(), occ0));
                    vertices.push(get_vertex(x + 1, y + 1, z + 0, block.color(), occ1));
                    vertices.push(get_vertex(x + 1, y + 1, z + 1, block.color(), occ2));
                    vertices.push(get_vertex(x + 1, y + 0, z + 1, block.color(), occ3));
                }
                
                if !surrounding[16] {
                    // South face
                    
                    //3 - 2
                    //| ╲ |
                    //0 - 1
                    
                    let occ0 = get_occlusion(surrounding[15], surrounding[24], surrounding[25]);
                    let occ1 = get_occlusion(surrounding[25], surrounding[26], surrounding[17]);
                    let occ2 = get_occlusion(surrounding[17], surrounding[8], surrounding[7]);
                    let occ3 = get_occlusion(surrounding[7], surrounding[6], surrounding[15]);
                    
                    indices.extend(&get_face(vertices.len() as u32, (occ0 - occ2).abs() < (occ1 - occ3).abs()));
                    
                    vertices.push(get_vertex(x + 0, y + 0, z + 0, block.color(), occ0));
                    vertices.push(get_vertex(x + 1, y + 0, z + 0, block.color(), occ1));
                    vertices.push(get_vertex(x + 1, y + 0, z + 1, block.color(), occ2));
                    vertices.push(get_vertex(x + 0, y + 0, z + 1, block.color(), occ3));
                }
                
                if !surrounding[12] {
                    // West face
                    
                    //3 - 2
                    //| ╲ |
                    //0 - 1
                    
                    let occ0 = get_occlusion(surrounding[9], surrounding[18], surrounding[21]);
                    let occ1 = get_occlusion(surrounding[21], surrounding[24], surrounding[15]);
                    let occ2 = get_occlusion(surrounding[15], surrounding[6], surrounding[3]);
                    let occ3 = get_occlusion(surrounding[3], surrounding[0], surrounding[9]);
                    
                    indices.extend(&get_face(vertices.len() as u32, (occ0 - occ2).abs() < (occ1 - occ3).abs()));
                    
                    vertices.push(get_vertex(x + 0, y + 1, z + 0, block.color(), occ0));
                    vertices.push(get_vertex(x + 0, y + 0, z + 0, block.color(), occ1));
                    vertices.push(get_vertex(x + 0, y + 0, z + 1, block.color(), occ2));
                    vertices.push(get_vertex(x + 0, y + 1, z + 1, block.color(), occ3));
                }
            }
        }
    }
    
    (vertices, indices)
}

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    position: [u8; 3],
    color: [u8; 4],
    occlusion: f32
}

implement_vertex!(Vertex, position, color, occlusion);
