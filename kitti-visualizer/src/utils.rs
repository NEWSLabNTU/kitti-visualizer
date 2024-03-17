use crate::{
    read_pcd::{load_bin, InfoPoint},
    PcdFormat,
};
use anyhow::{Context, Result};
use kitti_format::{KittiCalib, KittiObject};
use nalgebra as na;
use std::{fs, io, path::Path};

pub struct FrameData {
    pub objects: Vec<KittiObject>,
    pub points_in_range: Vec<InfoPoint>,
    pub points_out_range: Vec<InfoPoint>,
    pub num_points_map: Vec<usize>,
}

pub fn get_indices_from_ann_dir(ann_dir: &Path) -> Vec<usize> {
    let mut entries = fs::read_dir(ann_dir)
        .unwrap()
        .map(|res| res.map(|e| e.path()))
        .collect::<io::Result<Vec<_>>>()
        .unwrap();
    entries.sort();
    let indices: Vec<usize> = entries
        .iter()
        .filter_map(|path| {
            let path = path.as_path();
            let ext = path.extension();
            if ext.is_some() && ext.unwrap() == "txt" {
                let index_str = String::from(path.file_stem().unwrap().to_str().unwrap());
                let index = index_str.parse::<usize>().unwrap();
                Some(index)
            } else {
                None
            }
        })
        .collect();
    indices
}

pub fn in_bbox(point: &na::Point3<f64>, objects: &[KittiObject]) -> bool {
    let mut result = false;
    for obj in objects {
        let bbox = &obj.bbox3d;
        let enlarge_offset = 0.1;
        let origin_point = bbox.pose.inverse() * point;
        if origin_point.x < bbox.extents.x / 2. + enlarge_offset
            && origin_point.x > -bbox.extents.x / 2. - enlarge_offset
            && origin_point.y < bbox.extents.y / 2. + enlarge_offset
            && origin_point.y > -bbox.extents.y / 2. - enlarge_offset
            && origin_point.z < bbox.extents.z / 2. + enlarge_offset
            && origin_point.z > -bbox.extents.z / 2. - enlarge_offset
        {
            result = true;
            break;
        }
    }

    result
}

pub fn get_objects_from_frame_id(
    index: i32,
    kitti_dir: &Path,
    supervisely_ann_dir: Option<&Path>,
    pcd_format: PcdFormat,
) -> Vec<KittiObject> {
    let objects = if supervisely_ann_dir.is_none() {
        let exclude_classes = vec!["DontCare".into()];
        let ann_dir = kitti_dir.join("label_2");
        let calib_dir = kitti_dir.join("calib");
        let ann_path = ann_dir.join(format!("{:0>6}.txt", index.to_string()));
        let calib_path = calib_dir.join(format!("{:0>6}.txt", index.to_string()));

        let calib = KittiCalib::from_file(calib_path);

        if pcd_format == PcdFormat::Philly {
            kitti_format::read_ann_file_philly(ann_path, &calib, &exclude_classes)
        } else {
            kitti_format::read_ann_file(ann_path, &calib, &exclude_classes)
        }
    } else {
        let ann_path = supervisely_ann_dir
            .as_ref()
            .unwrap()
            .join(format!("{:0>6}.pcd.json", index.to_string()));
        let objects: Vec<KittiObject> = kitti_format::read_from_supervisely(&ann_path);
        objects
    };
    objects
}

pub fn get_new_frame_data(
    index: i32,
    kitti_dir: &Path,
    supervisely_ann_dir: Option<&Path>,
    pcd_format: PcdFormat,
) -> Result<FrameData> {
    let pcd_dir = kitti_dir.join("velodyne");
    let objects = get_objects_from_frame_id(index, kitti_dir, supervisely_ann_dir, pcd_format);
    // let objects = index_to_objects.get(&index.unwrap()).unwrap();
    // Get the pcd file
    let pcd_path = pcd_dir.join(format!("{:0>6}.bin", index.to_string()));
    let info_points =
        load_bin(&pcd_path).with_context(|| format!("unable to read {}", pcd_path.display()))?;
    let points_in_range: Vec<_> = info_points
        .iter()
        .filter(|p| {
            in_bbox(
                &na::Point3::from([p.point.x, p.point.y, p.point.z]).cast(),
                &objects,
            )
        })
        .map(|p| (*p).clone())
        .collect();
    let points_out_range: Vec<_> = info_points
        .iter()
        .filter(|p| {
            !in_bbox(
                &na::Point3::from([p.point.x, p.point.y, p.point.z]).cast(),
                &objects,
            )
        })
        .map(|p| (*p).clone())
        .collect();
    let num_points_map: Vec<usize> = objects
        .iter()
        .map(|obj| {
            let num_points = info_points
                .iter()
                .filter(|p| {
                    in_bbox(
                        &na::Point3::from([p.point.x, p.point.y, p.point.z]).cast(),
                        &[obj.clone()],
                    )
                })
                .count();
            num_points
        })
        .collect();

    Ok(FrameData {
        objects,
        points_in_range,
        points_out_range,
        num_points_map,
    })
}
