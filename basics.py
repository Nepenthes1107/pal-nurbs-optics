from enum import Enum
import torch
import numpy as np
import os
import datetime
import xlsxwriter as xw

torch.set_default_dtype(torch.float64)


class PrettyPrinter():
    def __str__(self):
        lines = [self.__class__.__name__ + ':']
        for key, val in vars(self).items():
            if val.__class__.__name__ in ('list', 'tuple'):
                for i, v in enumerate(val):
                    lines += '{}[{}]: {}'.format(key, i, v).split('\n')

            elif val.__class__.__name__ in 'dict':
                pass
            elif key == key.upper() and len(key) > 5:
                pass
            else:
                lines += '{}: {}'.format(key, val).split('\n')
        return '\n    '.join(lines)

    def to(self, device=torch.device('cpu')):
        for key, val in vars(self).items():
            if torch.is_tensor(val):
                exec('self.{x} = self.{x}.to(device)'.format(x=key))
            elif issubclass(type(val), PrettyPrinter):
                exec(f'self.{key}.to(device)')
            elif val.__class__.__name__ in ('list', 'tuple'):
                for i, v in enumerate(val):
                    if torch.is_tensor(v):
                        exec('self.{x}[{i}] = self.{x}[{i}].to(device)'.format(x=key, i=i))
                    elif issubclass(type(v), PrettyPrinter):
                        exec('self.{}[{}].to(device)'.format(key, i))


class Endpoint(PrettyPrinter):
    """
    Abstract class for objects.
    """

    def __init__(self, transformation, device=torch.device('cpu')):
        self.to_world = transformation
        self.to_object = transformation.inverse()
        self.device = device

    def intersect(self, ray):
        raise NotImplementedError()

    def sample_ray(self, position_sample=None):
        raise NotImplementedError()

    def draw_points(self, ax, options, seq=range(3)):
        raise NotImplementedError()

    def update_Rt(self, R, t):
        self.to_world = Transformation(R, t)
        self.to_object = self.to_world.inverse()
        self.to(self.device)


class Transformation(PrettyPrinter):
    """
    Rigid Transformation.

    - R is the rotation matrix.
    - t is the translational vector.
    """

    # JNS:先偏移在倾斜
    def __init__(self, R, t):
        if torch.is_tensor(R):
            self.R = R
        else:
            self.R = torch.Tensor(R)
        if torch.is_tensor(t):
            self.t = t
        else:
            self.t = torch.Tensor(t)

    def transform_point(self, o):
        o = torch.squeeze(self.R @ o[..., None]) + self.t
        if len(o.shape) == 2:
            o = torch.unsqueeze(o, dim=1)
        if len(o.shape) == 1:
            o = torch.unsqueeze(o, dim=0)
            o = torch.unsqueeze(o, dim=0)
        return o

    def transform_vector(self, d):
        d = torch.squeeze(self.R @ d[..., None])
        if len(d.shape) == 2:
            d = torch.unsqueeze(d, dim=1)
        if len(d.shape) == 1:
            d = torch.unsqueeze(d, dim=0)
            d = torch.unsqueeze(d, dim=0)
        return d

    def transform_ray(self, ray):
        o = self.transform_point(ray.o)
        d = self.transform_vector(ray.d)
        return Ray(o, d, ray.wavelength, ray.weight, ray.phase, device=o.device)

    def inverse(self):
        RT = self.R.T
        t = self.t
        return Transformation(RT, -RT @ t)


class Ray(PrettyPrinter):
    """
    Definition of a geometric ray.

    - o is the ray origin
    - d is the ray direction (normalized)
    """

    def __init__(self, o, d, wavelength=None, weight=None, phase=None, device=torch.device("cpu")):
        self.o = o
        self.d = d
        if phase is None:
            self.phase = 0
        else:
            self.phase = phase
        if weight is None:
            self.weight = torch.ones(o.shape[:-1]).to(device)
        else:
            self.weight = weight
        # scalar-version
        self.wavelength = wavelength  # [nm]
        self.mint = 1e-5  # [mm]
        self.maxt = 1e5  # [mm]
        self.to(device)

    def __call__(self, t):
        return self.o + t[..., None] * self.d


class Material(PrettyPrinter):
    """
    Optical materials for computing the refractive indices.

    The following follows the simple formula that

    n(λ) = A + B / λ^2

    where the two constants A and B can be computed from nD (index at 589.3 nm) and V (abbe number).
    """

    def __init__(self, name=None):
        self.name = 'vacuum' if name is None else name.lower()  # 小写化

        # This table is hard-coded. TODO: Import glass libraries from Zemax.
        self.MATERIAL_TABLE = {  # [nD, Abbe number]
            "vacuum": [1., np.inf],
            "air": [1.000000, np.inf],
            "e1": [1.3777, 50.23],
            "e2": [1.3371, 50.23],
            "e3": [1.3976, 50.23],
            "e4": [1.4033, 50.23],
            "e5": [1.3377, 50.23],
            "mr": [1.66875745552627, 32.],

            # SCHOTT.AGF
            "sf66": ['sellmeier_1', 2.078422330E+00, 1.808751340E-02, 4.071200320E-01, 6.794935720E-02, 1.767112920E+00,
                     2.152661270E+02],
            "sf15": ['sellmeier_1', 1.539259270E+00, 1.193079610E-02, 2.476209260E-01, 5.560775360E-02, 1.038164090E+00,
                     1.164167470E+02],
            "n-lak33a": ['sellmeier_1', 1.441169990E+00, 6.809338770E-03, 5.717495010E-01, 2.222918240E-02,
                         1.166052260E+00, 8.093795550E+01],
            "n-sf6": ['sellmeier_1', 1.779317630E+00, 1.337141820E-02, 3.381498660E-01, 6.175336210E-02,
                      2.087344740E+00, 1.740175900E+02],
            "n-lak33b": ['sellmeier_1', 1.422886010E+00, 6.702834520E-03, 5.936613360E-01, 2.194162100E-02,
                         1.161352600E+00, 8.074077010E+01],
            "n-laf36": ['sellmeier_1', 1.857442280E+00, 9.823971910E-03, 2.940987290E-01, 3.843091380E-02,
                        1.166154170E+00, 8.939846340E+01],
            "n-laf2": ['sellmeier_1', 1.809842270E+00, 1.017116220E-02, 1.572955500E-01, 4.424317650E-02,
                       1.093003700E+00, 1.006877480E+02],
            "k4": ['sellmeier_1', 1.189094860E+00, 7.893966420E-03, 8.413592140E-02, 3.392189950E-02, 9.302198220E-01,
                   1.070494950E+02],
            "laf11a": ['sellmeier_1', 1.668908170E+00, 1.138698980E-02, 3.119634710E-01, 5.003110480E-02,
                       9.768387850E-01, 8.732472660E+01],

            # BIREFRINGENT.AGF
            "calcite-e": ['sellmeier_1', 1.085600000E+00, 6.236260900E-03, 9.880000000E-02, 2.016400000E-02,
                          3.170000000E-01, 1.315150240E+02],
            # MISC.AGF
            "basf55": ['schott', 2.808085300E+00, -1.307651500E-02, 2.496132400E-02, 1.941273400E-03, -1.577674200E-04,
                       1.456295600E-05],
            "pmma": ['schott', 2.186458200E+00, -2.447534800E-04, 1.415578700E-02, -4.432978100E-04, 7.766425900E-05,
                     -2.993638200E-06],
            "pc": ['schott', 2.42838566, -3.86116645E-5, 2.8757447E-2, -1.97897366E-4, 1.48358968E-4, 1.38651935E-6],
            "polystyr": ['schott', 2.445983680E+00, 2.214289330E-05, 2.729885690E-02, 3.012108520E-04, 8.889348880E-05,
                         -1.757079290E-06],
            "polycarb": ['schott', 2.42838566E+00, -3.86116645E-05, 2.87574474E-02, -1.97897366E-04, 1.48358968E-04,
                         1.38651935E-06],


            # CDGM.AGF
            "h-lafl5": ['sellmeier_1', 1.265625170E+00, 1.026547970E+02, 1.813247700E+00, 1.116148700E-02,
                        1.735257910E-01, 5.413887640E-02],
            "h-laf3b": ['sellmeier_1', 1.664869690E+00, 8.956467120E-03, 3.016224800E-01, 3.502996950E-02,
                        1.197388800E+00, 1.233344380E+02],
            # OSAKAGASCHEMICAL.AGF
            "okp4": ['schott', 2.492299240E+00, -1.467131480E-03, 3.040591170E-02, -2.319597060E-04, 3.629283300E-04,
                     -1.891033190E-05],
            #  EYE.AGF
            "aqueous":['conrady', 1.32107278E+000, 8.47113739E-003, 2.31825063E-004],
            "lens":['conrady', 1.40146790E+000, 9.38901135E-003, 3.93175776E-004],
            "cornea":['conrady', 1.36313817E+000, 6.67127181E-003, 3.86916734E-004],
            "vitreous":['conrady', 1.32238376E+000, 6.72767909E-003, 3.33967702E-004],
            "mr7":['conrady', 1.60101964E+000, 3.91746260E-002, 8.91588562E-005],
            # other
            "grada": ['grin_3', 1.368, -0.001978, 0, 0],

        }
        self.dispersion = self._lookup_material()

    def ior(self, lam, x=0.0, y=0.0):
        """Computes index of refraction of a given wavelength (in [nm])"""
        e_type = self.dispersion[0]
        if e_type == "schott":
            a0, a1, a2, a3, a4, a5 = self.dispersion[1:]
            lam = lam * 1e-3  # 单位从nm转为um
            return self.schott(lam, a0, a1, a2, a3, a4, a5)

        elif e_type == 'sellmeier_1':
            K1, L1, K2, L2, K3, L3 = self.dispersion[1:]
            lam = lam * 1e-3  # 单位从nm转为um
            return self.sellmeier_1(lam, K1, L1, K2, L2, K3, L3)

        elif e_type == 'conrady':
            n0, A, B = self.dispersion[1:]
            lam = lam * 1e-3  # 单位从nm转为um
            return self.conrady(lam, n0, A , B)

        elif e_type == 'grin_3':
            raise TypeError(
                f"Material({self.name!r}) is a gradient medium: its index is a per-surface "
                "property. Use Gradient_3.axial_ior() / Gradient_3.get_ior() instead of "
                "Material.ior(). The material table cannot represent two surfaces with "
                "different base indices."
            )
        else:
            return self.buchdahl(lam, *self.dispersion)

    LAM_D_UM = 0.587562
    LAM_F_UM = 0.486133
    LAM_C_UM = 0.656273
    LAM_G_UM = 0.4358343

    @classmethod
    def _omega(cls, lam_um):
        """Buchdahl dispersion coordinate for wavelength in micrometres."""
        delta = lam_um - cls.LAM_D_UM
        return delta / (1.0 + 2.5 * delta)

    @classmethod
    def buchdahl(cls, lam, n_d, nu_1, nu_2):
        """Zemax model-glass dispersion; ``lam`` is in nm."""
        w = cls._omega(lam * 1e-3)
        return n_d + nu_1 * w + nu_2 * w ** 2

    @classmethod
    def nV_to_AB(cls, n, V):
        """Convert model-glass ``(n_d, V_d)`` to Buchdahl ``(n_d, nu1, nu2)``."""
        if V == 0 or not np.isfinite(V):
            return [n, 0.0, 0.0]
        delta_n = (n - 1.0) / V
        p_gf = 0.6438 - 0.001682 * V
        w_f = cls._omega(cls.LAM_F_UM)
        w_c = cls._omega(cls.LAM_C_UM)
        w_g = cls._omega(cls.LAM_G_UM)
        matrix = np.array(
            [[w_f - w_c, w_f ** 2 - w_c ** 2],
             [w_g - w_f, w_g ** 2 - w_f ** 2]],
            dtype=np.float64,
        )
        nu_1, nu_2 = np.linalg.solve(
            matrix, np.array([delta_n, p_gf * delta_n], dtype=np.float64)
        )
        return [n, float(nu_1), float(nu_2)]

    def _lookup_material(self):
        out = self.MATERIAL_TABLE.get(self.name)
        if isinstance(out, list):
            if type(out[0]) == str:
                return out
            else:
                n, V = out
        elif out is None:
            # try parsing input as a n/V pair
            tmp = self.name.split('/')
            n, V = float(tmp[0]), float(tmp[1])
        return self.nV_to_AB(n, V)

    def to_string(self):
        return f"{self.dispersion[0]} + {self.dispersion[1]}*omega + {self.dispersion[2]}*omega^2"

    def schott(self, lam, a0, a1, a2, a3, a4, a5):
        n_2 = a0 + a1 * lam ** 2 + a2 * lam ** (-2) + a3 * lam ** (-4) + a4 * lam ** (
            -6) + a5 * lam ** (-8)
        return torch.sqrt(n_2)

    def sellmeier_1(self, lam, K1, L1, K2, L2, K3, L3):
        n_2 = (K1 * lam ** 2 / (lam ** 2 - L1) + K2 * lam ** 2 / (lam ** 2 - L2) + K3 * lam ** 2 / (lam ** 2 - L3)) + 1
        return torch.sqrt(n_2)

    def conrady(self, lam, n0, A, B):
        n = n0 + A/lam + B/(lam ** 3.5)
        return n


def normalize(d):
    d1 = torch.sum(d ** 2, dim=-1)
    d2 = torch.sqrt(d1)
    return d / d2[..., None]


def rodrigues_rotation_matrix(k, theta):  # theta: [rad]
    """
    This function implements the Rodrigues rotation matrix.
    """
    # cross-product matrix
    kx, ky, kz = k[0], k[1], k[2]
    K = torch.Tensor([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0]
    ]).to(k.device)
    if not torch.is_tensor(theta):
        theta = torch.Tensor(np.asarray(theta)).to(k.device)
    return torch.eye(3, device=k.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K


def data_toExcel(lensdata, path):
    # xlsxwriter库储存数据到excel
    """
    path: The root directory of the file
    lens_sdata: It's a list with list element while the list element have string element
    """
    # 确保目录存在，如果不存在则创建目录
    if not os.path.exists(path):
        os.makedirs(path)

    # 获取当前时间并创建新文件的文件名
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = formatted_time + "lens_optimization.xlsx"

    # 构建文件的完整路径
    file_path = os.path.join(path, file_name)

    workbook = xw.Workbook(file_path)  # 创建工作簿
    worksheet1 = workbook.add_worksheet("sheet1")  # 创建子表
    worksheet1.activate()  # 激活表
    title = ['lens_optimization']  # 设置表头
    worksheet1.write_row('A1', title)  # 从A1单元格开始写入表头
    label = ['surface', 'thickness', 'roc', 'semi_diameter ', 'material ']
    worksheet1.write_row('A2', label)
    for i in range(len(lensdata)):
        row = 'A' + str(i + 3)  # start from A3
        worksheet1.write_row(row, lensdata[i])

    workbook.close()  # 关闭表
    print("New lensdata of xlsx created: ", file_path)


def rotate_anticlockwise(x, y, theta):
    """
    简单地实现坐标旋转
    theta为正，顺时针；theta为负，逆时针
    """
    # 将角度转换为弧度
    theta_rad = np.deg2rad(theta)

    # 计算旋转后的坐标
    x_rotated = x * np.cos(theta_rad) - y * np.sin(theta_rad)
    y_rotated = x * np.sin(theta_rad) + y * np.cos(theta_rad)

    return x_rotated, y_rotated
