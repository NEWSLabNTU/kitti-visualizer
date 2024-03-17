use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use nalgebra as na;
use std::{
    fs::File,
    io::{self, prelude::*, BufReader},
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
    let mut input = BufReader::new(
        File::open(pcd_path)
            .with_context(|| format!("Failed to open file {}", pcd_path.display()))?,
    );

    macro_rules! read_f32 {
        () => {{
            input.read_f32::<LittleEndian>()
        }};
    }

    macro_rules! try_read_f32 {
        () => {{
            let mut buf = [0u8; 4];
            let cnt = input.read(&mut buf)?;

            match cnt {
                4 => Ok(Some(f32::from_le_bytes(buf))),
                0 => Ok(None),
                cnt => Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("Truncated f32 found. Expect 4 bytes, but read {cnt} bytes."),
                )),
            }
        }};
    }

    let mut points = vec![];

    loop {
        let Some(x) = try_read_f32!()? else {
            break;
        };
        let y = read_f32!()?;
        let z = read_f32!()?;
        let intensity = read_f32!()?;

        let point = InfoPoint {
            point: [x, y, z].into(),
            intensity,
            device_id: None,
            active: None,
        };
        points.push(point);
    }

    Ok(points)
}
