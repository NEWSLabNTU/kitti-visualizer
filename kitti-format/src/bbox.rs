use nalgebra as na;

#[derive(Debug, Clone, PartialEq)]
pub struct BBox2D {
    pub t: f64,
    pub l: f64,
    pub h: f64,
    pub w: f64,
}

impl BBox2D {
    pub fn from_tlbr(tlbr: [f64; 4]) -> Self {
        let [t, l, b, r] = tlbr;
        Self {
            t,
            l,
            h: b - t,
            w: r - l,
        }
    }

    pub fn from_tlhw(tlhw: [f64; 4]) -> Self {
        let [t, l, h, w] = tlhw;
        Self { t, l, h, w }
    }

    pub fn tlhw(&self) -> [f64; 4] {
        let Self { t, l, h, w } = *self;
        [t, l, h, w]
    }

    pub fn tlbr(&self) -> [f64; 4] {
        let Self { t, l, h, w } = *self;
        [t, l, t + h, l + w]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BBox3D {
    pub extents: na::Vector3<f64>,
    pub pose: na::Isometry3<f64>,
}

impl BBox3D {
    pub fn vertex(&self, x_choice: bool, y_choice: bool, z_choice: bool) -> na::Point3<f64> {
        let point = {
            let x = self.extents.x / 2.0 * if x_choice { 1.0 } else { -1.0 };
            let y = self.extents.y / 2.0 * if y_choice { 1.0 } else { -1.0 };
            let z = self.extents.z / 2.0 * if z_choice { 1.0 } else { -1.0 };
            na::Point3::new(x, y, z)
        };
        self.pose * point
    }

    pub fn vertices(&self) -> Vec<na::Point3<f64>> {
        (0b000..=0b111)
            .map(|mask| self.vertex(mask & 0b001 != 0, mask & 0b010 != 0, mask & 0b100 != 0))
            .collect()
    }
}
