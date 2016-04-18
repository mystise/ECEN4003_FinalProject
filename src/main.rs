#[macro_use]
extern crate glium;
extern crate cgmath;
extern crate noise;
extern crate rand;

mod chunk;
mod position;
mod scheduler;

use std::time::{Instant, Duration, UNIX_EPOCH, SystemTime};
use std::env;

use glium::{Surface, glutin, DisplayBuild, Program};
use glium::glutin::{ElementState, Event, VirtualKeyCode};
use cgmath::{Matrix4, Point3, Vector3, Deg};

use scheduler::MasterScheduler;

fn main() {
    let display = glutin::WindowBuilder::new()
        .with_vsync()
        .with_depth_buffer(24)
        .with_dimensions(1024, 768)
        .with_title(format!("Parallel Voxels"))
        .build_glium()
        .unwrap();
    
    let program = Program::from_source(&display, "
    #version 140
    uniform mat4 model_view_projection;
    uniform vec2 chunk_position;
    in vec3 position;
    in vec4 color;
    in float occlusion;
    out vec4 v_color;
    void main() {
        gl_Position = model_view_projection * (vec4(chunk_position*16.0, 0.0, 0.0) + vec4(position, 1.0));
        v_color = color / 255.0 * occlusion;
    }
    ", "
    #version 140
    in vec4 v_color;
    out vec4 f_color;
    void main() {
        f_color = vec4(v_color.rgb, 1.0);
    }
    ", None).unwrap();
    
    let mut move_camera = true;
    let mut distant_camera = false;
    let mut camera_side_speed = 0.0;
    let mut camera_speed = 0.1;
    
    let mut testing_mode = 0;
    let mut camera_position = [0.5, 0.0, 0.0f32];
    let mut seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as u32;
    let mut thread_count = 4;
    
    for arg in env::args() {
        let arg = arg.split('=').collect::<Vec<&str>>();
        match arg[0] {
            "-s" => {
                seed = arg[1].parse().unwrap();
            },
            "-tc" => {
                thread_count = arg[1].parse().unwrap();
            },
            "-tm" => {
                testing_mode = 1;
                camera_speed = 16.0;
                distant_camera = true;
            }
            _ => {
                
            }
        }
    }
    
    let mut scheduler = MasterScheduler::new(seed, thread_count);
    
    let timestep: Duration = Duration::new(0, 16666667);
    let mut time = Duration::new(0, 0);
    let mut prev_time = Instant::now();
    
    'outer: loop {
        let now = Instant::now();
        time = now - prev_time + time;
        prev_time = now;
        
        while time > timestep {
            time = time - timestep;
            
            scheduler.update(camera_position, &display);
            
            if move_camera {
                camera_position[0] += camera_side_speed;
                camera_position[1] += camera_speed;
            }
            
            if !distant_camera {
                camera_position[2] = {
                    let pos = (camera_position[0].floor() as i32, camera_position[1].floor() as i32 & !3);
                    let height = {
                        *[scheduler.get_height([pos.0, pos.1 - 4]),
                        scheduler.get_height([pos.0, pos.1 - 3]),
                        scheduler.get_height([pos.0, pos.1 - 2]),
                        scheduler.get_height([pos.0, pos.1 - 1]),
                        scheduler.get_height([pos.0, pos.1]),
                        scheduler.get_height([pos.0, pos.1 + 1]),
                        scheduler.get_height([pos.0, pos.1 + 2]),
                        scheduler.get_height([pos.0, pos.1 + 3])].iter().max().unwrap() + 4
                    };
                    let next_height = {
                        *[scheduler.get_height([pos.0, pos.1]),
                        scheduler.get_height([pos.0, pos.1 + 1]),
                        scheduler.get_height([pos.0, pos.1 + 2]),
                        scheduler.get_height([pos.0, pos.1 + 3]),
                        scheduler.get_height([pos.0, pos.1 + 4]),
                        scheduler.get_height([pos.0, pos.1 + 5]),
                        scheduler.get_height([pos.0, pos.1 + 6]),
                        scheduler.get_height([pos.0, pos.1 + 7])].iter().max().unwrap() + 4
                    };
                    
                    let y_interp = (camera_position[1] - pos.1 as f32)/4.0;
                    
                    ((height as f32) * (1.0 - y_interp) + (next_height as f32) * y_interp).max(camera_position[2]-0.05)
                };
            } else {
                camera_position[2] = 128.0;
            }
            
            if testing_mode != 0 && camera_position[1] >= 4096.0 {
                match testing_mode {
                    1 => {
                        camera_position[1] = 0.0;
                        testing_mode = 2;
                        scheduler.serially_generate = !scheduler.serially_generate;
                        scheduler.serially_populate = !scheduler.serially_populate;
                        scheduler.serially_mesh = !scheduler.serially_mesh;
                    },
                    2 => {
                        break 'outer;
                    },
                    _ => {
                        unreachable!();
                    }
                }
            }
        }
        
        let (width, height) = {
            let window = display.get_window().unwrap();
            let size = window.get_inner_size_points().unwrap();
            (size.0 as f32, size.1 as f32)
        };
        
        let projection = cgmath::perspective(Deg {s: 90.0}, width / height, 0.01, 1000.0f32);
        
        let model_view = if distant_camera {
            Matrix4::look_at(Point3::from(camera_position), Point3::from(camera_position) + Vector3::new(0.0, 0.0, -1.0f32), Vector3::new(0.0, 1.0, 0.0f32))
        } else {
            Matrix4::look_at(Point3::from(camera_position), Point3::from(camera_position) + Vector3::new(0.0, 1.0, 0.0f32), Vector3::new(0.0, 0.0, 1.0f32))
        };
        
        let model_view_projection = projection * model_view;
        
        let mut frame = display.draw();
        frame.clear_color_and_depth((0.0, 0.2, 0.8, 0.0), 1.0);
        
        scheduler.render(&mut frame, &program, model_view_projection);
        
        frame.finish().unwrap();
        
        for event in display.poll_events() {
            match event {
                Event::Closed => return,
                Event::KeyboardInput(state, _, key_code) => {
                    if state == ElementState::Released {
                        if testing_mode != 0 {
                            continue;
                        }
                        if let Some(key_code) = key_code {
                            match key_code {
                                VirtualKeyCode::Key1 => {
                                    println!("Serial generation enabled");
                                    scheduler.serially_generate = true;
                                },
                                VirtualKeyCode::Key2 => {
                                    println!("Parallel generation enabled");
                                    scheduler.serially_generate = false;
                                },
                                VirtualKeyCode::Key3 => {
                                    println!("Serial population enabled");
                                    scheduler.serially_populate = true;
                                },
                                VirtualKeyCode::Key4 => {
                                    println!("Parallel population enabled");
                                    scheduler.serially_populate = false;
                                },
                                VirtualKeyCode::Key5 => {
                                    println!("Serial meshing enabled");
                                    scheduler.serially_mesh = true;
                                },
                                VirtualKeyCode::Key6 => {
                                    println!("Parallel meshing enabled");
                                    scheduler.serially_mesh = false;
                                },
                                VirtualKeyCode::Space => {
                                    move_camera = !move_camera;
                                },
                                VirtualKeyCode::P => {
                                    distant_camera = !distant_camera;
                                    camera_position[2] = 0.0;
                                },
                                VirtualKeyCode::Minus => {
                                    camera_speed -= 0.1;
                                },
                                VirtualKeyCode::Equals => {
                                    camera_speed += 0.1;
                                },
                                VirtualKeyCode::LBracket => {
                                    camera_side_speed -= 0.1;
                                },
                                VirtualKeyCode::RBracket => {
                                    camera_side_speed += 0.1;
                                },
                                VirtualKeyCode::Semicolon => {
                                    camera_side_speed = 0.0;
                                    camera_speed = 0.0;
                                },
                                VirtualKeyCode::Escape => {
                                    break 'outer;
                                },
                                _ => ()
                            }
                        }
                    }
                },
                _ => ()
            }
        }
    }
    
    println!("SG: {:.2} ms PG: {:.2} ms", scheduler.serial_generator_time, scheduler.parallel_generator_time);
    println!("SP: {:.2} ms PP: {:.2} ms", scheduler.serial_populator_time, scheduler.parallel_populator_time);
    println!("SM: {:.2} ms PM: {:.2} ms", scheduler.serial_meshing_time, scheduler.parallel_meshing_time);
}
