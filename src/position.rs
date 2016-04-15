#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub struct ChunkPosition {
    pub x: i32,
    pub y: i32
}

impl ChunkPosition {
    pub fn north(&self) -> ChunkPosition {
        ChunkPosition {x: self.x, y: self.y + 1}
    }
    
    pub fn north_east(&self) -> ChunkPosition {
        ChunkPosition {x: self.x + 1, y: self.y + 1}
    }
    
    pub fn east(&self) -> ChunkPosition {
        ChunkPosition {x: self.x + 1, y: self.y}
    }
    
    pub fn south_east(&self) -> ChunkPosition {
        ChunkPosition {x: self.x + 1, y: self.y - 1}
    }
    
    pub fn south(&self) -> ChunkPosition {
        ChunkPosition {x: self.x, y: self.y - 1}
    }
    
    pub fn south_west(&self) -> ChunkPosition {
        ChunkPosition {x: self.x - 1, y: self.y - 1}
    }
    
    pub fn west(&self) -> ChunkPosition {
        ChunkPosition {x: self.x - 1, y: self.y}
    }
    
    pub fn north_west(&self) -> ChunkPosition {
        ChunkPosition {x: self.x - 1, y: self.y + 1}
    }
}

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub struct Position {
    pub x: i32,
    pub y: i32,
    pub z: i32
}

impl Position {
    /*pub fn offset(&self, dir: Direction) -> Position {
        let mut output = self.clone();
        
        match dir {
            Direction::Up => {
                output.z += 1;
            },
            Direction::Down => {
                output.z -= 1;
            },
            Direction::North => {
                output.y += 1;
            },
            Direction::South => {
                output.y -= 1;
            },
            Direction::East => {
                output.x += 1;
            },
            Direction::West => {
                output.x -= 1;
            }
        }
        
        output
    }*/
}

/*#[derive(Copy, Clone, Eq, PartialEq)]
pub enum Direction {
    Up,
    Down,
    North,
    South,
    East,
    West
}*/
