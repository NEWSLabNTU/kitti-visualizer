mod gui;
mod read_pcd;
mod utils;

use crate::gui::Gui;
use anyhow::Result;
use clap::{Parser, ValueEnum};
use kiss3d::window::Window;
use std::path::PathBuf;

#[derive(Parser)]
struct Opts {
    #[clap(short, long)]
    pub kitti_dir: PathBuf,
    #[clap(short, long)]
    pub supervisely_ann_dir: Option<PathBuf>,
    #[clap(short, long, default_value = "libpcl")]
    pub format: PcdFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
pub enum PcdFormat {
    Libpcl,
    Philly,
}

fn main() -> Result<()> {
    let Opts {
        kitti_dir,
        supervisely_ann_dir,
        format,
    } = Opts::parse();

    visualize_in_pcd(kitti_dir, supervisely_ann_dir, format)?;

    Ok(())
}

fn visualize_in_pcd(
    kitti_dir: PathBuf,
    supervisely_ann_dir: Option<PathBuf>,
    pcd_format: PcdFormat,
) -> Result<()> {
    let mut window = Window::new_with_size("debug", 1920, 1080);
    window.set_background_color(1., 1., 1.);
    window.set_line_width(2.);

    let gui = Gui::new(kitti_dir, supervisely_ann_dir, pcd_format)?;
    window.render_loop(gui);

    Ok(())
}

// fn visualize_in_image(kitti_dir: PathBuf) {
//     let exclude_classes = vec!["DontCare".into()];
//     let image_dir = kitti_dir.join("image_2");
//     let calib_dir = kitti_dir.join("calib");
//     let ann_dir = kitti_dir.join("label_2");
//     let indices = get_indices_from_ann_dir(&ann_dir);
//     let mut idx = 0;
//     let mut index: Option<i32> = Some(indices[idx]);
//     let mut mode = DrawingMode::Draw3D;

//     loop {
//         if index.is_none() {
//             eprint!("Enter the dataset index: ");
//             index = Some(read!());
//             idx = indices.iter().position(|&i| Some(i) == index).unwrap();
//         }
//         // Get the calib file
//         let calib_path = calib_dir.join(format!("{:0>6}.txt", index.unwrap().to_string()));
//         let calib = KittiCalib::from_file(calib_path);
//         // Get the annotation file
//         let ann_path = ann_dir.join(format!("{:0>6}.txt", index.unwrap().to_string()));
//         let objects = kitti_format::read_ann_file(ann_path, &calib, &exclude_classes);
//         let image_path = image_dir.join(format!("{:0>6}.png", index.unwrap().to_string()));
//         let mut image = imread(image_path.to_str().unwrap(), 1).unwrap();

//         // Draw bboxes
//         draw_objects(&mut image, &objects, &calib, &mode);
//         put_text(
//             &mut image,
//             &format!(
//                 "frame index: {}",
//                 format!("{:0>6}", index.unwrap().to_string())
//             ),
//             cv_core::Point::new(0, 30),
//         );
//         imshow("image", &image).unwrap();
//         loop {
//             let key = wait_key(0).unwrap();
//             if key == 27 {
//                 index = None;
//                 break;
//             }
//             if key == 81 {
//                 if idx == 0 {
//                     idx = indices.len() - 1;
//                 } else {
//                     idx = idx - 1;
//                 }

//                 index = Some(indices[idx]);
//                 break;
//             }
//             if key == 83 {
//                 idx = idx + 1;
//                 if idx >= indices.len() {
//                     idx = 0;
//                 }
//                 index = Some(indices[idx]);
//                 break;
//             }
//             if key == 115 {
//                 match mode {
//                     DrawingMode::Draw3D => {
//                         mode = DrawingMode::Draw2D;
//                     }
//                     DrawingMode::Draw2D => {
//                         mode = DrawingMode::Draw3D;
//                     }
//                 }
//                 break;
//             }
//             // if key == 114 {
//             //     objects
//             //         .iter_mut()
//             //         .for_each(|obj| (*obj).bbox3d.rot_y += 0.1);
//             //     break;
//             // }
//         }
//     }
// }
