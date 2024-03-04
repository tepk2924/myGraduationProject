#!/usr/bin/env python3

from ast import Mod
from logging import raiseExceptions
import numpy as np
import trimesh
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
import potpourri3d as pp3d
import yaml
import os
import inspect

from typing import Optional
from scipy.interpolate import splprep, splev
from scipy import interpolate
from scipy import optimize
from scipy.integrate import trapz
from numpy import gradient

import suction_cup_functions as scf

# Params
# --------------------------------------------------
# 40mm cup: r=17.6, d_0=60,k_bm = 11*10**-3, k_def = 2.9892, coef_friction = 0.5, lip_area = 450, per_smoothing = 0, k_n smoothing = 1,moment_multiplicator = 4,


class ModelData():
    """
    Mesh 정보와 Suction Cup의 매개변수를 저장하는 클래스
    ----------------------------------------------------
    """

    r = 17.6 #유효 반경 (mm)
    d_0 = 60
    max_deformation = 10 #최대 변형 (mm)
    k_bm = 1.1e-2  # Coefficient for curvature
    k_def = 2.9892 # Coefficient for deformation pressure
    coef_friction = 0.5
    lip_area = 450 #컵의 입술 부분 넓이 (mm^2)
    perimiter_smoothing = 1
    per_points = 60

    config_dict = {
        "r": r,
        "d_0": d_0,
        "max_deformation": max_deformation,
        "k_bm": k_bm,
        "k_def": k_def,
        "coef_friction": coef_friction,
        "lip_area": lip_area,
        "perimiter_smoothing": perimiter_smoothing,
        "per_points": per_points
    }

    def __init__(self,
                 mesh: trimesh.base.Trimesh, 
                 load_path: Optional[str] = None,
                 *args,
                 **kwargs):
        """
        매개변수
        ----------
        mesh_location : str
            Mesh 파일 경로 (obj, stl, ply, etc.)
        load_path : Optional[str], optional
            Config 파일의 경로. 기본값 None
        scalar : float
            Mesh의 크기를 변환하는 배수값

        키워드 변수
        ----------
        units : tuple(str, str)
            A tuple containing the units of the mesh, and the units we want to convert to. IE ("meters", "millimeters")
            Suction cup model has only been tested with meshes in millimeters. !!!!!
        subdivide : bool
            If true, the mesh will be subdivided. This is useful for low resolution meshes.
        """

        #Mesh의 Watertightness 검사. 만약 Watertightness하지 않으면, 고의로 에러를 발생.
        self.mesh = mesh

        if "units" in kwargs:   # Convert mesh to correct units
            if kwargs["units"][1] == "meters":
                print("WARNING: Suction cup model has only been tested with meshes in millimeters.")
            self.mesh.units = kwargs["units"][0]
            self.mesh.convert_units(kwargs["units"][1])
        if "subdivide" in kwargs:   # Subdivide the mesh
            if kwargs["subdivide"] == True:
                self.mesh = self.mesh.subdivide()
        
        if load_path is not None:
            self.load_config(load_path)

        self.intersection_mesh = trimesh.ray.ray_triangle.RayMeshIntersector(
            self.mesh)
        self.proximity_mesh = trimesh.proximity.ProximityQuery(self.mesh)

        #Mesh의 표면에서 30000개의 지점 샘플링
        self.samples, self.faces = trimesh.sample.sample_surface(
            self.mesh, 30000)
        self.heatmap = False

    def save_config(self, path, name):
        with open(os.path.join(path, name) + ".yml", "w") as yaml_file:
            yaml.dump(self.config_dict, yaml_file, default_flow_style=False)
        print("Saved config:")
        print(self.config_dict)

    def load_config(self, path):
        with open(path, "r") as yaml_file:
            config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
            for attr in list(config_dict.keys()):
                if not attr.startswith("__"):
                    setattr(ModelData, attr, config_dict[attr])
  
        print("Loaded config:")
        print(config_dict)

    def create_heat_map(self):
        self.pp3d_solver = pp3d.PointCloudHeatSolver(self.samples)
        self.tangent_frames = self.pp3d_solver.get_tangent_frames()
        self.heatmap = True

    def model_transform(self, tf):
        self.mesh = self.mesh.apply_transform(tf)
        self.intersection_mesh = trimesh.ray.ray_triangle.RayMeshIntersector(
            self.mesh)
        self.proximity_mesh = trimesh.proximity.ProximityQuery(self.mesh)


class SuctionContact():
    """
    주어진 지점에 대해 역학 분석과 밀봉 형성 유무 검사하는 클래스
    -------------------------------------------------------------
    """
    def __init__(self, con_p):
        """
        주어진 지점에 대해 역학 분석과 밀봉 형성 유무 검사하는 클래스
        -------------------------------------------------------------
        매개변수
        --------------
        con_p : np.array(3,)
            - Point on the model we want to contact
        """
        self.con_p = con_p

    def form_seal(self, model: ModelData):
        """
        모델의 접촉 둘레를 구해서 저장하고, n_avg를 구하는 부분이 포함되어 있음. 성공 여부가 self.success에 저장됨.
        --------------
        model : ModelData
        """
        if model.heatmap == False:
            model.create_heat_map()

        # try:
        per_success = self.get_perimiter(model)
        # except:
        #per_success = False
        if per_success:
            self.success = True
            self.calculate_along_path()
            self.tangent, self.radial = self.calculate_tangents(model)
            self.calculate_average_normal()
        else:
            self.success = False

    def get_perimiter(self, model: ModelData, max_deviation: float = 2, min_prox: float = 1):
        """
        self.con_p에서 model.r만큼의 지오데식 거리를 가지는 지점들 리턴
        --------------
        model : ModelData
        max_deviation : float
            허용 오차. con_p에서 지오데식 거리가 model.r로부터 max_deformation만큼 최대 오차를 허용
        min_prox : float
            A filter parameter for searching the points. We discard the found points that are inside the radius = min_prox
        Returns
        --------------
        suceess : bool
            True if the perimiter was successfully found and false otherwise
        """

        #샘플링한 30000개 지점들 불러오기
        faces = model.faces
        samples = model.samples

        #샘플링한 점들 중 con_p에서 가장 가까운 지점을 p_0점으로 선정
        self.p_0_i = scf.closest_node(self.con_p, samples, 1)
        self.p_0 = samples[self.p_0_i]
        cloasest, distance, triangle_id = model.proximity_mesh.on_surface([
                                                                          self.p_0])
        self.p_0_n = model.mesh.face_normals[triangle_id[0]]

        #다른 모든 지점에 대해 지오데식 거리 계산
        dist = model.pp3d_solver.compute_distance(self.p_0_i)

        #허용 가능 지오데식 거리 범위 설정 후 실제 그 범위에 들어가는 지점들만 걸러내기
        perimeter_min = model.r - max_deviation
        perimeter_max = model.r + max_deviation
        smaller = np.where(dist < perimeter_max)
        bigger = np.where(dist > perimeter_min)
        intersection = np.intersect1d(smaller, bigger)
        points_filtered = np.array(samples[intersection])

        #그러한 지점이 없으면 실패 리턴 후 메소드 조기 종료.
        if len(points_filtered) == 0:
            return False
            # raise Exception(
            #     "No points found in the interest area. Probably an invalid object.")

        #이 perimiter의 지점들은 원래 생성될 때 무작위 위치의 30000개의 점에서 model.r의 일정 허용 오차 범위에서 골라온 것.
        #즉, 지오데식 거리는 model.r에서 최대 max_deviation까지 거리 오차가 나는 상황이므로,
        #이를 보정하기 위해 지점들을 옮기는 작업을 수행하는 것.
        v_per_p0 = self.p_0-points_filtered
        # Get normals at edge points
        normals_at_per = model.tangent_frames[2][intersection]
        # Projection of v_per_p0 to the tangent plane
        temp = (v_per_p0*normals_at_per).sum(1)
        #v_tangent는 perimiter 각 지점에서 접평면에 놓여있으면서, 방향은 대충 p_0을 향하게 된다.
        v_tangent = v_per_p0-temp[:, np.newaxis]*normals_at_per
        v_tangent = scf.unit_array_of_vectors(v_tangent)
        #거리를 계산하고 이를 바탕으로 실제 지점이 움직여야 하는 거리를 계산
        points_deviation = dist[intersection]-model.r
        #실제 지점을 옯기는 코드
        v_tangent = v_tangent*points_deviation[:, np.newaxis]
        points_filtered = points_filtered + v_tangent
        #움직인 지점들을 현재 mesh에 투영.
        points_filtered, distance, faces_filtered = model.proximity_mesh.on_surface(
            points_filtered)
        #너무 가까우면 제외
        points_filtered, mask = trimesh.points.remove_close(
            points_filtered, radius=min_prox)

        #걸러낸 지점들에 대해 순서 부여하기
        #외판원 순회 문제로 해결. (항상 올바르지는 않음)
        order_of_points, dist_eucl = trimesh.points.tsp(
            points_filtered, start=0)
        #order_of_points = np.append(order_of_points, order_of_points[0])
        #극좌표를 이용하는 방법이 있지만, 여기서는 사용하지 않음
        normals_at_per = np.array(model.mesh.face_normals[faces_filtered])
        average_normal_approx = scf.unit_vector(
            np.average(normals_at_per, axis=0))
        self.average_normal_approx = average_normal_approx
        #order_of_points = scf.radial_sort(points_filtered, self.p_0, average_normal_approx)
        #order_of_points = np.append(order_of_points, order_of_points[0])

        #순서가 부여된 접촉 둘레. 개수가 5보다 작으면 실패 판정 후 메소드 조기 종료
        self.perimiter = np.array(points_filtered[order_of_points])
        if len(self.perimiter) < 5:
            return False

        self.normal = self._get_normals(
            self.perimiter, model)
        self.calculate_along_path()
        self.tangent, self.radial = self.calculate_tangents(model)

        #무언가를 4번까지 시도해서 조건 만족 안 하면 실패 처리하고 메소드 조기 종료. 뭐지?
        for _ in range(4):
            test_dir = (
                self.radial*scf.unit_array_of_vectors(self.p_0-self.perimiter)).sum(1)
            if not (test_dir > 0).all():
                if len(self.perimiter) < 5:
                    return False
                majority = np.average(test_dir)
                if majority < 0:
                    self.perimiter = np.flip(self.perimiter, 0)
                    self.normal = self._get_normals(
                        self.perimiter, model)
                    self.calculate_along_path()
                    self.tangent, self.radial = self.calculate_tangents(
                        model)
                else:
                    # Select the point to delete
                    tangent_projections = np.dot(
                        self.tangent, np.roll(self.tangent, -1, axis=0).T)
                    tangent_projections = np.diagonal(tangent_projections)
                    min_1, min_2 = np.argpartition(test_dir, 2)[:2]

                    if min_1 > min_2:
                        min_1, min_2 = min_2, min_1

                    self.perimiter = np.delete(self.perimiter, min_1, axis=0)
                    self.normal = np.delete(self.normal, min_1, axis=0)
                    self.calculate_along_path()
                    self.tangent, self.radial = self.calculate_tangents(model)

                    # Rerun tsp
                    order_of_points, dist_eucl = trimesh.points.tsp(self.perimiter, start=0)
                    self.perimiter = np.array(self.perimiter[order_of_points])

            else:
                break
        else:
            #print("Can not obtain a valid perimiter for the point.")
            return False


        #접촉 둘레 내삽 과정. model.perpoint개만큼의 지점을 선정하여 perimeter_pnts로 저장.
        self.perimiter = np.append(
            self.perimiter, self.perimiter[np.newaxis, 0, :], axis=0)

        self.perimiter_unfiltered = self.perimiter
        if len(self.perimiter_unfiltered) < 5:
            return False
        per_func = self.interpolate_perimiter(self.perimiter, model.perimiter_smoothing)
        u = np.linspace(0, 1, model.per_points)
        perimiter_pnts = per_func(u)

        #perimeter의 각 지점 perimeter_pnts의 접선, 법선, 반경 벡터 구하기
        self.normal = self._get_normals(
            perimiter_pnts, model)
        self.perimiter = np.array(perimiter_pnts)
        self.calculate_along_path()
        self.tangent, self.radial = self.calculate_tangents(model)

        #모든 과정 성공시 self.perimeter, self.normal, self.tangent, self.radial에 model.perpoint개만큼의 정보가 저장되고, 성공을 반환하여 메소드 정상 종료.
        return True

    def _get_normals(self, points, model, use_barycentric=True):
        """
        접촉 둘레 내삽 과정에서 해당 지점들은 법선벡터가 계산되어 있던 30000개의 지점이 아니므로, 따로 법선 벡터를 계산하는 과정이 필요.
        --------------
        points : np.array(n,3)
            법선 벡터 계산 필요 지점들
        model : ModelData
        use_barycentric : bool
            무게중심 좌표계 사용 내삽 유무
        Returns
        --------------
        None : None
        """
        closest, distance, triangle_id = model.proximity_mesh.on_surface(
            points)

        if use_barycentric:
            # Normal at final point
            bary = trimesh.triangles.points_to_barycentric(
                model.mesh.triangles[triangle_id], closest, method='cramer')
            bary = np.array(bary)
            face_points = model.mesh.faces[triangle_id]
            normals = np.asarray(model.mesh.vertex_normals[face_points])

            normal = (bary[:, :, np.newaxis]*normals).sum(1)
            normal = scf.unit_array_of_vectors(normal)

        else:
            normal = model.mesh.face_normals[triangle_id]
        return np.array(normal)

    def calculate_along_path(self):
        """
        u와 du 함수 도출 메소드
        --------------
        locations : (n, 3) float
            Ordered locations of contact points
        Returns
        --------------
        """
        locations = self.perimiter
        distance = np.roll(locations, -1, axis=0) - locations
        distance = np.linalg.norm(distance, axis=1)
        distance_sum = np.sum(distance)
        self.du = distance/distance_sum
        self.du_cumulative = np.cumsum(np.append(0, self.du[:-1]))

    def calculate_tangents(self, model):
        """
        접선, 법선 벡터 계산. calculate_along_path 메소드가 이 메소드 실행 전에 선행되어야 함.
        --------------
        locations : (n, 3) float
            Ordered array of contact locations
        normal : (n, 3) float
            An array of normals to the contact locations
        Returns
        --------------
        tangent : (n, 3) float
        An array of tangets to the path along contact locations
        radial : (n, 3) float
        A cross product of normal and tangent
        """

        locations = self.perimiter
        normal = self.normal
        tangent = gradient(locations, self.du_cumulative, axis=0, edge_order=2)
        norm = np.linalg.norm(tangent, axis=1)

        tangent[:, 0] = np.divide(
            tangent[:, 0], norm)
        tangent[:, 1] = np.divide(
            tangent[:, 1], norm)
        tangent[:, 2] = np.divide(
            tangent[:, 2], norm)

        radial = np.cross(normal, tangent)

        return tangent, radial

    def interpolate_perimiter(self, perimeter_pnts, smoothing):
        """
        30000개의 지점 중에서 거리 조건을 만족시키고 순서도 부여된 지점들을 바탕으로 내삽하는 과정.
        --------------------
        """
        x = perimeter_pnts[:, 0]
        y = perimeter_pnts[:, 1]
        z = perimeter_pnts[:, 2]

        tck_per, u = splprep([x, y, z], s=smoothing, per=True, k=3)

        def tck_per_func(u):
            new_p = splev(u, tck_per)
            return np.transpose(new_p)

        return tck_per_func

    #이 함수는 여기 정의되기만 하고 이 파일 내에서는 따로 사용되지는 않음. 외부에서 불러서 쓰는 용도인 듯.
    def interpolate_normal(self, normal_pnts, smoothing):
        n_x = normal_pnts[:, 0]
        n_y = normal_pnts[:, 1]
        n_z = normal_pnts[:, 2]

        tck_nor, u = splprep([n_x, n_y, n_z], s=smoothing, per=True)

        def tck_nor_func(u):
            new_n = splev(u, tck_nor)
            new_n = new_n/np.linalg.norm(new_n, axis=0, keepdims=True)
            return np.transpose(new_n)

        return tck_nor_func
    
    #이 함수는 여기 정의되기만 하고 이 파일 내에서는 따로 사용되지는 않음. 외부에서 불러서 쓰는 용도인 듯.
    def interpolate_tangent(self, tangent_pnts, smoothing):
        t_x = tangent_pnts[:, 0]
        t_y = tangent_pnts[:, 1]
        t_z = tangent_pnts[:, 2]

        tck_x = scipy.interpolate.splrep(
            self.du_cumulative, t_x, s=smoothing, per=True)
        tck_y = scipy.interpolate.splrep(
            self.du_cumulative, t_y, s=smoothing, per=True)
        tck_z = scipy.interpolate.splrep(
            self.du_cumulative, t_z, s=smoothing, per=True)

        def tck_tan_func(u):
            new_n_x = splev(u, tck_x)
            new_n_y = splev(u, tck_y)
            new_n_z = splev(u, tck_z)
            new_n = np.array([new_n_x, new_n_y, new_n_z]).T
            temp = np.linalg.norm(new_n, axis=1)
            new_n = new_n/np.linalg.norm(new_n, axis=1, keepdims=True)
            return new_n

        return tck_tan_func

    def calculate_average_normal(self):
        """
        self.normal의 값으로부터 n_avg 계산 후 self.averagr_normal에 저장.
        -------
        """
        #함수 이름을 보아서는 사다리꼴 방법을 쓰는 것으로 보임.
        average_normal_x = trapz(
            self.normal[:, 0], self.du_cumulative)
        average_normal_y = trapz(
            self.normal[:, 1], self.du_cumulative)
        average_normal_z = trapz(
            self.normal[:, 2], self.du_cumulative)
        average_normal = np.array(
            [average_normal_x, average_normal_y, average_normal_z])
        #정규화.
        average_normal = average_normal/np.linalg.norm(average_normal)

        self.average_normal = average_normal

    def find_apex(self, a_v, model: ModelData):
        """
        컵의 꼭지점의 좌표 도출.
        -----------------
        """
        # Find the "lowest perimeter point"
        # projection of point to approach normal
        dist = np.dot(self.perimiter, np.transpose(a_v))
        # index of a point that is furthest away
        max_dist_i = int(dist.argmin())

        # We must find root of this function. That we can get poisition of apex

        def find_a(t, approach, point_mi):
            dist = np.linalg.norm(
                self.p_0 - approach*t - point_mi) - (model.d_0-model.max_deformation)
            return dist

        t_0 = optimize.root(find_a, model.d_0, args=(
            a_v, self.perimiter[max_dist_i, :]))

        a = -a_v*t_0.x + self.p_0  # Position of apex

        return a

    def _calculate_deformation(self, a_v, model, per_points):
        """
        논문에서 언급된 d(u)함수 계산하는 부분
        ----------------
        """
        # Deformation Using projection on Approach vector
        dist = np.dot(per_points, np.transpose(a_v))
        max_dist_i = int(dist.argmin())
        distance = dist-dist[max_dist_i]-model.max_deformation
        return distance

    def _calculate_deformation_vectors(self, a_v):
        """
        Returns
        ---------------
        [[-a_v],
        [-a_v],
        ...
        [-a_v]]
        """
        #def_vectors = unit_array_of_vectors(self.apex - self.perimiter)
        #def_vectors = self.normal
        def_vectors = -np.tile(a_v, (np.shape(self.normal)[0], 1))
        return def_vectors

    def evaluate_contact(self, a_v, model: ModelData, debug_display=False):
        """
        주어진 접근 벡터에 대해 밀봉 형성 여부 판단
        --------------
        a_v: (3, )  np.array
            Direction how the "robot" approaches the contact point. Usually same as -con_n
        model : ModelData class
            class containing the information of the mesh and suction cup properties.
        Returns
        --------------
        Seal success : bool
            True if seal formed and false otherwise.
        """

        apex = self.find_apex(a_v, model)
        self.apex = apex
        #d(u) 계산
        distance = self._calculate_deformation(a_v, model, self.perimiter)
        # Normal Curvature ------------------------------
        dx = 1/np.shape(self.tangent)[0]
        #self.tangent과 self.normal의 고주파수 영역을 제거하는 코드.
        #고주파수 영역을 남기면 미분 계산 때 부정확해서 그런 것인가?
        tangent_fit = scf.unit_array_of_vectors(
            scf.fourier_fit_3d(self.du_cumulative, self.tangent, 5))
        normal_fit = scf.unit_array_of_vectors(
            scf.fourier_fit_3d(self.du_cumulative, self.normal, 5))

        #곡률 계산
        #normal = (1/k)*(d(tanent)/ds)
        #k * normal = d(tangent)/ds
        #k * normal dotp normal = (d(tangent)/ds) dotp normal
        #k = (d(tangent)/ds) dotp normal
        ddg = np.gradient(tangent_fit, dx, axis=0, edge_order=2)
        k_n_rough = (ddg * normal_fit).sum(1)  # Curvature
        #스플라인 곡선으로 내삽
        try:
            tck = scipy.interpolate.splrep(
                self.du_cumulative, k_n_rough, s=1, per=False, k=3)

        except:
            return False
        k_n = splev(self.du_cumulative, tck)

        #논문 eq.7 계산
        self.p_bm = np.gradient(np.gradient(model.k_bm * k_n, dx,
                                            edge_order=2), dx, edge_order=2)


        # Presure because of deformation
        def_vectors = self._calculate_deformation_vectors(a_v)

        #논문 eq.3 계산
        p_d = (def_vectors.T * distance * model.k_def).T
        # Amount of pressure in direction of normal to surface
        p_d_n = (p_d * normal_fit).sum(1)

        # Analyzing the all pressures
        #밀봉 형성 조건 2번째에서 나오는 f_def 계산.
        self.p_all = p_d_n - self.p_bm

        p_max_i = np.argmax(self.p_all)

        # ------- PLOT FOR EASIER DETERMINING OF PARAMETERS ----------
        if debug_display:
            plt.subplot(3, 1, 1)
            plt.plot(self.du_cumulative, self.normal, "r", label="raw [x,y,z]")
            plt.plot(self.du_cumulative, normal_fit,
                     "g", label="fitted [x,y,z]")
            plt.title("Fitting normal using FFT")
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.plot(k_n_rough, "r", label="k_n before smoothing")
            plt.plot(k_n, "g", label="k_n after smoothing")
            plt.title("Smoothing the Curvature")
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(p_d_n, label="normal deformation pressure")
            plt.plot(self.p_bm, label="bending pressure")
            plt.plot(self.p_all, label="all together")
            plt.legend()
            plt.title("Pressure distribution")
            plt.show()
        # -------------------------------------------------------------

        if self.p_all[p_max_i] > 0:
            #밀봉 형성 조건 2번째가 만족되지 않음
            return False
        else:
            #밀봉 형성 조건 2번째가 만족됨.
            return True

    def evaluate_forces(self, f_pos, f_dir, m_vec, vacuum, model, a_v, in_current_configuration=False, simulate_object_rotation=True):
        """
        Returns a set of points and normals that are at a distance of r along the surfaceof the mesh
        --------------
        f_pos : 힘이 작용하는 지점 : (x,y,z) in mm.
        f_dir : 힘 벡터. (3,) array
        m_vec : 외부 모멘트 벡터 (3,)
        vacuum : 내외부 압력차 단위 : N/mm^2 (MPa)
        model : Object model that contains suction cup parameters
        in_current_configuration : Bool
            If True we will take the current rotation as the zero position. That means we the moment applied in this position will be considered as our zero point.
            If False we wont care about the current rotation of the object. The rotation of the object will be adjusted so the moment equals to external applied moment.
        simulate_object_rotation : Bool
            If True the external force will be rotated along with the object.
        --------------
        contact_succes : Returns true if the contact can sustain the force acting on it. False otherwise.
        """
        # {C} 좌표계 구축: x_ax, y_ax, z_ax는 각각 논문에서의 {C}좌표계의 축에 해당.
        dir_t = self.perimiter[0, :] - self.p_0
        #z_ax는 논문에서 언급한 대로 n_avg와 동일.
        z_ax = self.average_normal
        y_ax = np.cross(z_ax, dir_t)
        y_ax = y_ax/np.linalg.norm(y_ax)
        x_ax = np.cross(y_ax, z_ax)

        #계산 과정에서 클래스 내부 변수를 보호하기 위해 값 복사
        normal_cp = np.copy(self.normal)
        tangent_cp = np.copy(self.tangent)
        radial_cp = np.copy(self.radial)
        perimiter_cp = np.copy(self.perimiter)

        #외부 토크와 힘에 의한 돌림힘
        r_m = f_pos - self.p_0  #mm
        m = np.cross(r_m, f_dir) + m_vec  #N*mm

        #진공력---------------------------------------------
        #논문 eq.8, P_proj(u) 구하는 과정.
        proj_points = trimesh.points.project_to_plane(self.perimiter,
                                                      plane_normal=self.average_normal, plane_origin=[
                                                          0, 0, 0],
                                                      transform=None, return_transform=False, return_planar=True)
        #논문 eq.9에서 area(P_proj(u)) 구하는 식.
        area = scf.poly_area(proj_points[:, 0], proj_points[:, 1])

        # Compensate for moment ------------------------
        #모든 접촉둘레의 지점을 p_0이 원점이 되도록 평행이동.
        T = scf.translation(-self.p_0)
        perimiter_transpozed = np.ones((4, np.shape(self.perimiter)[0]))
        perimiter_transpozed[0:3, :] = perimiter_cp.T
        perimiter_transpozed = np.dot(T, perimiter_transpozed)

        dist = np.dot(perimiter_transpozed[0:3, :].T, np.transpose(a_v))
        offset = dist[int(dist.argmin())]
        distance = dist-offset-model.max_deformation

        # Presure because of deformation
        #이걸 또 계산하네?
        #eq.3 계산
        def_vectors = self._calculate_deformation_vectors(a_v)
        p_d = (def_vectors.T * distance * model.k_def).T

        #이번에는 축 방향으로 분해한다.
        def_n = -(p_d*self.normal).sum(1)
        def_t = -(p_d*self.tangent).sum(1)
        def_r = -(p_d*self.radial).sum(1)

        #f_com에 의한 모멘트 계산
        moment_sum = self.moment_calc(
            def_n, def_t, def_r, normal_cp, tangent_cp, radial_cp, np.transpose(perimiter_transpozed[0:3, :]))
        if in_current_configuration == True:
            m += moment_sum

        # -------------TEMP-----------------
        dir_t = self.perimiter[0, :] - self.p_0
        z_ax_2 = -a_v
        y_ax_2 = np.cross(z_ax_2, dir_t)
        y_ax_2 = y_ax_2/np.linalg.norm(y_ax_2)
        x_ax_2 = np.cross(y_ax_2, z_ax_2)

        f_zzz = self.force_calc(def_n+self.p_bm, def_t, def_r,
                                self.normal, self.tangent, self.radial, z_ax_2)
        f_xxx = self.force_calc(def_n+self.p_bm, def_t, def_r,
                                self.normal, self.tangent, self.radial, x_ax_2)
        f_yyy = self.force_calc(def_n+self.p_bm, def_t, def_r,
                                self.normal, self.tangent, self.radial, y_ax_2)

        self.deformation_force_mom = [
            np.array([f_xxx, f_yyy, f_zzz]), moment_sum]

        #논문의 F챕터의 첫 부분인 외부 모멘트와 컵에 의한 모멘트를 맞추기 위해 물체를 회전시키는 구간으로 추정된다만,
        #이상하게 논문의 내용과 사뭇 다른 계산식이 들어가 있는 것 같다.
        # Calculate the main moment axis-----------------
        #모멘트를 {C}좌표계 축으로 분리
        m_x = x_ax*(m*x_ax).sum()
        m_y = y_ax*(m*y_ax).sum()
        m_z = z_ax*(m*z_ax).sum()
        # Add together  x and y
        m_xy = m_x + m_y
        rot_modificator = 8000
        rot_sum = np.zeros(3)
        for i in range(50):
            #f_com에 의한 모멘트 계산
            moment_sum = self.moment_calc(
                def_n, def_t, def_r, normal_cp, tangent_cp, radial_cp, np.transpose(perimiter_transpozed[0:3, :]))

            #{C} 좌표계로 분리
            ma_x = x_ax*(moment_sum*x_ax).sum()
            ma_y = y_ax*(moment_sum*y_ax).sum()
            ma_z = z_ax*(moment_sum*z_ax).sum()
            # Compare with what the moment should be
            ma_xy = ma_x+ma_y
            #외부 힘에 의한 모멘트와 f_com에 의한 모멘트와 비교
            if np.linalg.norm(m_xy-ma_xy) < 0.1:
                break

            #논문에서의 r_ax?
            m_xy_dir = (m_xy-ma_xy)/np.linalg.norm(m_xy-ma_xy)

            if not np.allclose(ma_xy, m_xy, atol=40, rtol=0.001):

                rot_scale = np.sum(np.abs(ma_xy-m_xy))

                rotation = rot_scale/rot_modificator

                if rot_modificator < 8000:
                    rot_modificator += 1000

                # Rotate everything and try again #rot_scale*0.00007
                rot_sum += m_xy_dir*rotation
                #m_xy_dir (=r_ax)를 축으로 하고 rotation 만큼 회전시키는 회전행렬 생성
                R_mat = scf.rot_axis_angle(m_xy_dir, rotation)
                #좌표축 회전
                x_ax = np.dot(R_mat, x_ax)
                y_ax = np.dot(R_mat, y_ax)
                z_ax = np.dot(R_mat, z_ax)
                #접촉 둘레 상의 접,법선 및 반경 벡터 회전
                normal_cp = np.dot(R_mat, normal_cp.T).T
                tangent_cp = np.dot(R_mat, tangent_cp.T).T
                radial_cp = np.dot(R_mat, radial_cp.T).T
                #접촉 둘레를 이루는 지점들 회전
                perimiter_transpozed[0:3, :] = np.dot(
                    R_mat, perimiter_transpozed[0:3, :])
                #오브젝트 회전 후 주요 값들 재계산
                if simulate_object_rotation == True:
                    r_m = np.dot(R_mat, r_m)
                    m = np.cross(r_m, f_dir) + m_vec  # N*mm
                    # Calculate the main moment axis-----------------
                    # First we projct the moment to the contact coordinate System
                    m_x = x_ax*(m*x_ax).sum()
                    m_y = y_ax*(m*y_ax).sum()
                    m_z = z_ax*(m*z_ax).sum()
                    # Add together  x and y
                    m_xy = m_x + m_y

                # Recalculate distances
                dist = np.dot(
                    perimiter_transpozed[0:3, :].T, np.transpose(a_v))
                distance = dist-offset-model.max_deformation
                # Presure because of deformation
                # HERE  A KOEFFICIENT 4 FOR DEF. ANY OTHER WAY TO SOLVE THIS?????
                p_d = (def_vectors.T * distance * model.k_def).T
                # Projecting the deformation pressure
                def_n = -(p_d*normal_cp).sum(1)
                def_t = -(p_d*tangent_cp).sum(1)
                def_r = -(p_d*radial_cp).sum(1)
            else:
                self.perimiter_transpozed = perimiter_transpozed[0:3, :]
                #print(np.linalg.norm(temp))
                break
           
        if np.linalg.norm(rot_sum) >= 0.9:
            return False

        # z 방향 이송
        # In the end we add the moment around z axis to the pressure distribution
        m_z = ma_z-m_z
        # Distances of points to "z" axis that goes trough origin.
        z_ax_stacked = np.tile(z_ax, (np.shape(perimiter_cp)[0], 1))
        leavers_vec = np.transpose(perimiter_transpozed[0:3, :]) - (
            z_ax_stacked.T * (np.transpose(perimiter_transpozed[0:3, :])*z_ax).sum(1)).T
        leavers = np.linalg.norm(leavers_vec, axis=1)
        # We determine the moment each point has to provide
        m_z_spread = np.linalg.norm(m_z)  # *self.du
        # We calculate the force/pressure at each point
        m_z_p = -m_z_spread/leavers
        # We determine the directions in which the pressure acts
        m_z_p_dir = np.cross(z_ax, leavers_vec)
        m_z_p_dir = scf.unit_array_of_vectors(m_z_p_dir)
        # FInal moment around z axis as a vector distribution of pressure
        m_z_p_vec = (m_z_p_dir.T*m_z_p).T
        m_z_p_n = -(m_z_p_vec*normal_cp).sum(1)
        m_z_p_t = -(m_z_p_vec*tangent_cp).sum(1)
        m_z_p_r = -(m_z_p_vec*radial_cp).sum(1)

        #진공력 F_vac
        vac_n = np.tile(model.lip_area*vacuum, np.shape(normal_cp)[0])

        # Next we analyze the plane forces ----------------------------------------
        p_nor = def_n + m_z_p_n + vac_n
        p_tan = def_t + m_z_p_t
        p_rad = def_r + m_z_p_r

        # Actual force in the direction of average_normal
        def_n, def_t, def_r = 0, 0, 0
        self.premik = 0

        scale_coef = 3.5
        for i in range(10):
            # Calculate force
            force = self.force_calc(
                p_nor, p_tan, p_rad, normal_cp, tangent_cp, radial_cp, z_ax)
            force_sum = area*vacuum-force
            # What kind of force is desired in the direction of the average_normal
            force_desired = np.dot(-f_dir, z_ax)

            # We have reached equilibrium force, break the loop
            if np.allclose([force_sum], [force_desired], atol=1) == True:
                break
            else:
                # Transform the points up or down to get cloaser to the desired force
                sign = np.sign(force_sum - force_desired)
                scale = np.abs(force_sum - force_desired)

                self.premik += sign*z_ax_2*scale/3
                scale_coef += 3.5-0.3

                if i == 9:
                    print("WARNING FORCE NOT BALANCING")

                T_mat = scf.translation(sign*z_ax_2*scale/3.5)
                perimiter_transpozed[0:4, :] = np.dot(
                    T_mat, perimiter_transpozed)

                dist = np.dot(
                    perimiter_transpozed[0:3, :].T, np.transpose(a_v))
                distance = dist-offset-model.max_deformation

                def_vectors = self._calculate_deformation_vectors(a_v)
                # Presure because of deformation
                p_d = (def_vectors.T * distance * model.k_def).T
                # Projecting the deformation pressure
                def_n = -(p_d*normal_cp).sum(1)
                def_t = -(p_d*tangent_cp).sum(1)
                def_r = -(p_d*radial_cp).sum(1)
                p_nor = def_n + m_z_p_n + vac_n
                p_tan = m_z_p_t + def_t
                p_rad = m_z_p_r + def_r

        self.perimiter_transpozed = perimiter_transpozed[0:3, :]

        # Lastly add the pressure form the x and y axis forces
        f_x_already = self.force_calc(
            p_nor, p_tan, p_rad, normal_cp, tangent_cp, radial_cp, x_ax)
        f_y_already = self.force_calc(
            p_nor, p_tan, p_rad, normal_cp, tangent_cp, radial_cp, y_ax)
        f_x = (np.dot(-f_dir, x_ax)+f_x_already)*x_ax
        f_y = (np.dot(-f_dir, y_ax)+f_y_already)*y_ax
        f_x_n = (normal_cp*f_x).sum(1)
        f_x_t = (tangent_cp*f_x).sum(1)
        f_x_r = (radial_cp*f_x).sum(1)
        f_y_n = (normal_cp*f_y).sum(1)
        f_y_t = (tangent_cp*f_y).sum(1)
        f_y_r = (radial_cp*f_y).sum(1)

        p_nor += f_x_n + f_y_n
        p_tan += f_x_t + f_y_t
        p_rad += f_x_r + f_y_r

        # We also add the curvature pressure
        premik = np.linalg.norm(self.premik)
        reduction_p_bm = np.abs(premik-model.max_deformation)
        p_nor += self.p_bm

        #논문 eq.14 부등식의 양변.
        t1 = trapz(p_nor*model.coef_friction, self.du_cumulative)
        t2 = trapz(np.sqrt(p_tan**2+p_rad**2), self.du_cumulative)

        if p_nor[np.argmin(p_nor)] < 0:
            #print("Failure because of normal force.")
            return False
        if t2 > t1: #논문 eq.14 부등식 만족 여부 판단.
            #print("Failure bacause of friction force.")
            return False
        else:
            return True

    def force_calc(self, p_nor, p_tan, p_rad, n, t, r, direction):
        """
        Based on the given perimiter distributed force, force direction and perimiter points centered around 1
         the function calculates the force generated by the distributed force.
        --------------
        p_nor: (n, ) np.array
            Normal component of the distributed force
        p_tan: (n, ) np.array
            Tangent component of the distributed force
        p_rad: (n, ) np.array
            Radial component of the distributed force
        n: (n, 3) np.array
            Matrix containing the normals to the surface along the perimiter
        t: (n, 3) np.array
            Matrix containing the tangents to the surface along the perimiter
        r: (n, 3) np.array
            Matrix containing the radials to the surface along the perimiter
        perimiter: (n, 3) np.array
            Perimiter points centered around (0,0,0)
        --------------
        force : (3, ) np.array
            Force vector calculated given the inputs.
        """
        p_nor_v = (n.T * p_nor).T
        p_nor_p = (direction*p_nor_v).sum(1)
        force_n = trapz(p_nor_p, self.du_cumulative)

        p_tan_v = (t.T * p_tan).T
        p_tan_p = (direction*p_tan_v).sum(1)
        force_t = trapz(p_tan_p, self.du_cumulative)

        p_rad_v = (r.T * p_rad).T
        p_rad_p = (direction*p_rad_v).sum(1)
        force_r = trapz(p_rad_p, self.du_cumulative)

        return force_n+force_t+force_r

    def moment_calc(self, p_nor, p_tan, p_rad, n, t, r, perimeter):
        """
        Based on the given perimiter distributed force, force direction and perimiter points centered around 1
         the function calculates the moments generated by the distributed force.
        --------------
        p_nor: (n, ) np.array
            Normal component of the distributed force
        p_tan: (n, ) np.array
            Tangent component of the distributed force
        p_rad: (n, ) np.array
            Radial component of the distributed force
        n: (n, 3) np.array
            Matrix containing the normals to the surface along the perimiter
        t: (n, 3) np.array
            Matrix containing the tangents to the surface along the perimiter
        r: (n, 3) np.array
            Matrix containing the radials to the surface along the perimiter
        perimiter: (n, 3) np.array
            Perimiter points centered around (0,0,0)
        --------------
        moment : (3, ) np.array
            Moment vector calculated given the inputs.
        """

        p_nor_v = (n.T * p_nor).T
        p_tan_v = (t.T * p_tan).T
        p_rad_v = (r.T * p_rad).T

        pressure_sum = p_nor_v+p_tan_v+p_rad_v
        # Calculating the applied moment. We must transform everything to mm
        inside_vector = perimeter  # - np.array(self.p_0)
        moment = np.cross(inside_vector, pressure_sum) * \
            self.du[:, np.newaxis] * 4
        # We get a 3D moment in Nmm, We transform it to N*m
        moment_sum = np.sum(moment, axis=0)

        moment_x = moment_sum[0]
        moment_y = moment_sum[1]
        moment_z = moment_sum[2]

        return np.array([moment_x, moment_y, moment_z])
