use anyhow::Result;
use itertools::Itertools;
use nalgebra as na;
use std::{
    fs::File,
    io::{prelude::*, BufReader},
    path::Path,
};

#[derive(Clone, Debug)]
pub struct InfoPoint {
    pub point: na::Point3<f32>,
    pub intensity: f32,
    pub device_id: Option<u64>,
    pub active: Option<u64>,
}

pub fn load_bin(pcd_path: &Path) -> Result<Vec<InfoPoint>> {
    let mut input = BufReader::new(File::open(pcd_path).expect("Failed to open file"));
    let mut floats: Vec<f32> = vec![];
    loop {
        use std::io::ErrorKind;
        let mut buffer = [0u8; std::mem::size_of::<f32>()];
        let res = input.read_exact(&mut buffer);
        match res {
            Err(error) if error.kind() == ErrorKind::UnexpectedEof => break,
            _ => {}
        }
        res.expect("Unexpected error during read");
        let f = f32::from_le_bytes(buffer);
        floats.push(f);
    }
    let mut points: Vec<InfoPoint> = vec![];

    for (x, y, z, intensity) in floats.iter().tuples() {
        points.push(InfoPoint {
            point: na::Point3::from([*x, *y, *z]),
            intensity: intensity.clone(),
            device_id: None,
            active: None,
        });
    }

    Ok(points)
}
