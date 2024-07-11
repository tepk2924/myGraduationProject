#!/usr/bin/env python3

import trimesh
import time
import numpy as np
import scipy.spatial as ss
import multiprocessing
import traceback

from typing import List, Tuple, Union, Optional, Dict, Any
from scipy.spatial import cKDTree
from trimeshVisualize import Scene
from multiprocessing import Pool
import multiprocessing_utils

import common1_2_suction_cup_lib as sclib
import common1_3_suction_cup_functions as scf

# BELOW ARE THE FUNCTIONS FOR EVALUATIONG THE WHOLE MODEL
class EvaluateMC():
    def __init__(self,
                 obj_model: sclib.ModelData,
                 n_processors: int = 8,
                 number_of_points: int = 500,
                 point_sample_radius: float = 2.5,
                 multiprocessing: bool = True,
                 neighboor_average: int = 5,
                 noise_samples: int = 10,
                 noise_cone_angle: float = 0.2,
                 **kwargs) -> None:
        """
        Class used to evaluate grasps on an object
        --------------
        - Args:
            - obj_model {ModelData} 
                Object that contains the mesh and the parameters for the "Continious Suction Cup Model"
            - n_processors {int} : (default=8)
                Number of threads we will use.
            - number_of_points {int} : (default=500)
                Number of points we will sample on the object.
            - point_sample_radius {float} : (default=2.5)
                How close can the sampled points be to eachother.
            - multiprocessing {bool} : (default=True)
                Whether we will use multiprocessing or not.
            - neighboor_average {int} : (default=5)
                Average the score of neighboring N points
            - noise_samples {int} : (default=10)
                When evaluating the seal sample N approach vectors and evaluate binnary score for each sample.
            - noise_cone_angle {float} : (default=0.2)
                Angle of the cone used to sample the approach vectors.
        - Kwargs:
            - evaluate_force_method : "binary_search" for using binary search algorithm
                                      None or else for using constant shift

        """
        # Store set up parameters for
        self.obj_model = obj_model
        self.n_processors = n_processors
        self._number_of_points = number_of_points
        self._point_sample_radius = point_sample_radius
        self._multiprocessing = multiprocessing
        self._neighboor_average = neighboor_average
        self._noise_samples = noise_samples
        self._noise_cone_angle = noise_cone_angle
        if "evaluate_force_method" in kwargs:
            self.evaluate_force_method = kwargs["evaluate_force_method"]
        else:
            self.evaluate_force_method = "const"

    def evaluate_one_point_MP(self,
                              inp_vec: Tuple[np.ndarray, int],
                              evaluate_for_force: bool = True
                              ) -> List:
        """
        Evaluate one point on the object
        --------------
        inp_vec : (np.array(3,), int)
            Vector containing the contact point and face ID.
        evaluate_for_force : Bool (default = True)
            Perform force evaluation or just seal evaluation.
        --------------
        evaluation : [np.array(n,3), np.array(n,3), bool, float]
            list containing (p_0, p_0_n, contact_score, force_score)
        """
        con_p = inp_vec[0]  #샘플링 된 지점
        con_n = self.obj_model.mesh.face_normals[inp_vec[1]]

        #a_v 방향과 대강 비슷한 방향의 접근 벡터를 가정하였을 때, seal이 얼마나 큰 성공률로 형성이 되었는가?
        suction_contact, contact_score, a_v = contact_test_seal(
            con_p, self.obj_model, noise_samples=self._noise_samples, noise_cone_angle=self._noise_cone_angle)

        if contact_score == -1: #해당 con_p에서 form_seal에서부터 실패한 경우
            return [suction_contact.p_0, suction_contact.p_0_n, a_v, 0., 0.]
            
        if contact_score > 0: #해당 con_p에서 form_seal이 성공하였음
            if evaluate_for_force:
                #force_score : grasp wrench hull의 부피 (단위 : N^3) -> 얼마나 많은 범위의 외력을 견딜 수 있는가?
                #force_score 자체의 물리적인 의미는 희미하지만, 단순 지표로서 역할을 한다.
                force_score = contact_test_forces(
                    suction_contact, self.obj_model, evaluate_force_method=self.evaluate_force_method)
            else:
                force_score = 1 #이건 뭐지
            return [suction_contact.p_0, suction_contact.p_0_n, a_v, contact_score,  force_score]
        else:
            return [suction_contact.p_0, suction_contact.p_0_n, a_v, contact_score,  0.]

    def evaluate_model(self, display: bool = False) -> Dict:
        start = time.time()

        #self._number_of_points개 만큼 Mesh에서 지점 샘플링
        samples, face_id = trimesh.sample.sample_surface_even(
            self.obj_model.mesh, self._number_of_points, self._point_sample_radius)

        if self._multiprocessing:
            out = multiprocessing_utils.run_multiprocessing(self.evaluate_one_point_MP,
                                            zip(samples, face_id), self.n_processors)
        else:
            out = []
            for i, test_sample in enumerate(zip(samples, face_id)):
                out.append(self.evaluate_one_point_MP(test_sample))
                if i % 100 == 0:
                    print("Sameple: ", i)

        # Output performance
        print("Input length: {}".format(len(samples)))
        print("Output length: {}".format(len(out)))
        print("Multiprocessing time: {}mins".format((time.time()-start)/60))
        print("Multiprocessing time: {}secs".format((time.time()-start)))

        out_formated, out_formated_old = self._format_output(out)

        if display:
            print("Displaying scene")
            my_scene = Scene()
            #my_scene.plot_point_multiple(samples, radius=1)
            my_scene.plot_mesh(self.obj_model.mesh)
            #아래 코드는 제대로 출력하지 못해서 주석 처리함
            #ids = my_scene.plot_grasp(out_formated["tf"], out_formated["scores"])
            #대신, trimeshVisualize 코드에서 직접 코드 가져와서 수정함
            ids = []
            for i in range(len(out_formated["tf"])):
                if round(out_formated["scores"][i], 4) == 0:
                    continue
                grasp_point = np.array([0, 0, 0])
                #일단 10mm의 화살표로 처리.
                grasp_dir = np.array([0, 0, 10])
                points_transformed = trimesh.transform_points(
                    [grasp_point, grasp_dir], out_formated["tf"][i])
                grasp_point = np.array(points_transformed[0])
                grasp_dir = np.array(points_transformed[1])
                id = my_scene.plot_vector(grasp_point, grasp_dir, radius_cyl=1, arrow=True)
                ids.append(id)
            print(f"Trying to render {len(ids)} grasps found....")
            #좌표축 표시. 각 화살표는 100mm.
            my_scene.plot_vector(np.array([0, 0, 0]), np.array([100, 0, 0]),
                color=[255, 0, 0, 255], radius_cyl=1)
            my_scene.plot_vector(np.array([0, 0, 0]), np.array([0, 100, 0]),
                color=[0, 255, 0, 255], radius_cyl=1)
            my_scene.plot_vector(np.array([0, 0, 0]), np.array([0, 0, 100]),
                color=[0, 0, 255, 255], radius_cyl=1)
            my_scene.display()

        return out_formated

    def _format_output(self,
                       out: List
                       ) -> Tuple[Dict, Dict]:
        """
        Format the output from the evaluation
        --------------
        - out {list}
            -Output must have the following format:
            -[p_0, p_0_n, a_v, contact_success,  force_score]
        -----------
        - Returns:
            - formated_grasps {dict}
                - {"tf": [np.array(4, 4), ...], "scores": [float, ...]}
                - Output formated and converted into grasp tf's with scores
            - formated_output {dict}
                - {"p_0": [np.array(3,), ...],
                   "p_0_n": [np.array(3,), ...],
                   "a_v": [np.array(3,), ...],
                   "score_seal": [bool, ...],
                   "score_force": [float, ...],
                   "score_total": [float, ...]
                   }
                - Less formated output. More usefull for debugging
        """
        # Covert to numpy array of objects
        out = np.array(out, dtype=object)
        # Prepare a dict where we will store the formated output:
        final_output = {}
        ##########
        # STEP 1 #
        ##########

        # Remove all entries where contact_success is -1 (could not calculate something)
        out_formated = np.copy(out[out[:, 3] != -1])
        # Split out the points
        final_output["p_0"] = np.array(out_formated[:, 0])
        # Split out the normals
        final_output["p_0_n"] = np.array(out_formated[:, 1])
        # Split out the approach vector
        final_output["a_v"] = np.array(out_formated[:, 2])
        # Split out the seal score
        final_output["score_seal"] = np.array(out_formated[:, 3])
        # Split out the force score
        #반영되는 force score는 실제 N^3값이 아니라 (값/최댓값)으로 반영한다.
        if out_formated[np.argmax(out_formated[:, 4]), 4] != 0:
            final_output["score_force"] = np.array(
                out_formated[:, 4]) / out_formated[np.argmax(out_formated[:, 4]),4]
        else:
            final_output["score_force"] = np.zeros(out_formated[:, 4].shape)

        ##########
        # STEP 2 #
        ##########

        #Contact_score = score_force * score_seal
        #즉, 해당 지점에서 a_v에 노이즈를 주었을 때의 성공률 * wrench hull의 부피 / wrench hull의 부피의 최댓값
        #항상 0 ~ 1의 값을 가진다.
        point_total_score = final_output["score_force"] * \
            final_output["score_seal"]
        final_output["score_total"] = point_total_score


        if self._neighboor_average > 1:
            #k-d 트리: k차원 공간의 점들을 구조화하는 공간 분할 자료구조.
            kdtree = cKDTree(list(final_output["p_0"]))
            dist, points = kdtree.query(
                list(final_output["p_0"]), self._neighboor_average)
            point_total_score = np.where(point_total_score[np.array(points)[:,0]] > 0, np.average(
                point_total_score[np.array(points)], axis=1), 0)
                
        formated_grasps = {"tf":[], "scores":[]}
        
        for i, pnt_score in enumerate(point_total_score):
            if pnt_score > 0:
                if type(final_output["p_0"][i]) == None:
                    continue 
                grasp_tf = trimesh.geometry.align_vectors(
                    np.array([0, 0, -1]), final_output["a_v"][i])
                grasp_tf[0:3, 3] = final_output["p_0"][i]
                formated_grasps["tf"].append(grasp_tf)
                formated_grasps["scores"].append(pnt_score)

        formated_grasps["tf"] = np.array(formated_grasps["tf"])
        formated_grasps["scores"] = np.array(formated_grasps["scores"])
        return formated_grasps, final_output


def suction_force_limits(file_loc: str,
                         con_p: np.ndarray,
                         force_direction: Optional[np.ndarray] = None,
                         vac_min: float = 0.020,
                         vac_max: float = 0.065,
                         increment: float = 0.005,
                         ) -> np.ndarray:
    """ Test an object for suction force limits

    Parameters
    ----------
    file_loc : str
        File location of the object
    con_p : np.ndarray
        Contact point coordinates. Cloasest point on the mesh from  this point will be used as the contact point
    force_direction : Optional[np.ndarray], optional
        The direction in which the pull away force acts. If none the surface normal is used, by default None
    vac_min : float, optional
        Minimum vacuum level to test for, by default 0.020
    vac_max : float, optional
        Maximum vacuum level to test for, by default 0.065
    increment : float, optional
        At what vacuum increment to test for, by default 0.005

    Returns
    -------
    np.ndarray
        A (N, 2) array containing the vacuum level and the force at that vacuum level

    Raises
    ------
    Exception
        If the suction contact fails to initialize. This can happen if the perimeter can not be found (do not know why this happens).
    """
    

    obj_model = sclib.ModelData(file_loc)
    # Initiate contact
    suction_contact = sclib.SuctionContact(con_p)

    # Form seal
    suction_contact.form_seal(obj_model)
    if suction_contact.success == False:
        raise Exception("Failed to get the perimeter")

    a_v = -suction_contact.average_normal

    if force_direction is None:
        force_direction = a_v
    
    contact_success = suction_contact.evaluate_contact(a_v, obj_model)
    vacuums = np.arange(vac_min, vac_max, increment)

    results = []

    for vacuum_level in vacuums:
        contact_success = True
        force = 0
        while contact_success:
            contact_success = suction_contact.evaluate_forces(suction_contact.p_0, a_v*force, np.array([0,0,0]), vacuum_level, obj_model, a_v, in_current_configuration = False, simulate_object_rotation = True)

            force += 0.2

        results.append((vacuum_level, force))

    return np.array(results)

def contact_test_forces(suction_contact: sclib.SuctionContact,
                        obj_model: sclib.ModelData,
                        vac_level: float = 0.07,
                        **kwargs) -> float:
    """
    Test the given suction contact for resistance to external forces. The score is the volume of "wrench space"
    --------------
    - Args:
        - suction_contact {SuctionContact}
            - A suction contact that has a seal already formed
        - obj_model {ModelData()}
            - Data about our cup, gripper, ...
    - Kwargs:
        - vac_level {float} : Default=0.07 MPa
            - The vacuum level for which to test the contact
        - kwargs : {"a_v":np.array(3,)}
            - "a_v" specfic approach vector for a contact. If none is given -average_normal is used.
    --------------
    - Returns:
        - convex_hull.volume {float}
            - The volume of the convex hull
    """
    #물체의 무게 중심?
    cog = obj_model.mesh.center_mass

    force_location = cog
    external_moment = np.array([0, 0, 0])

    if "a_v" in kwargs:
        a_v = kwargs["a_v"]
    else:
        #입력 안 받으면 일단 n_avg의 반대방향 벡터로 설정.
        a_v = -suction_contact.average_normal

    if "evaluate_force_method" in kwargs:
        method = kwargs["evaluate_force_method"]
    else:
        method = "const"
    #print(f"method : {method}")

    force_0 = (cog-suction_contact.p_0) / \
        np.linalg.norm(cog-suction_contact.p_0)
    #z성분이 양수인 단위벡터 몇 개 생성
    force_directions = scf.create_half_sphere()

    #(0, 0, 1)벡터에서 a_v로 변환하는 회전행렬
    R_mat = trimesh.geometry.align_vectors(np.array([0, 0, 1]), a_v)
    #force_directions에 해당 변환 적용
    force_directions = np.dot(R_mat[0:3, 0:3], force_directions.T).T
    #이 결과로, force direction은 a_v와 대강 방향이 비슷한 (a_v와 dotp시 양수가 나오는) 단위 벡터 여러 개가 됨.
    results = []

    if method == "const":
        #print("2N만큼 조절 방법")
        for force in force_directions:
            i = 18
            #먼저, force방향으로 18N, 물체의 무게중심에 작용하는 것으로 테스트
            success_prev = suction_contact.evaluate_forces(
                force_location, force*i, external_moment, vac_level, obj_model, a_v)
            success = success_prev

            #일단 i값을 바꾸는 알고리즘을 단순히 2N씩 증감으로 해놓았는데, 개선할 여지가 있다.
            #이분 탐색을 실행하면 더 넓은 범위에서 더 빠르게 한계 i값을 찾을 수 있지 않을까?
            while success == success_prev:
                if success:
                    i += 2
                else:
                    i -= 2
                
                #단, 배율을 못 찾아서 80N 이상의 힘까지 올라가는 상황이거나 음수로 내려가는 상황이라면 전체 실패로 간주.
                if (i > 80):
                    print(i, "KAY SE DOGAJA")

                    return 0.
                elif (i < 0):
                    return 0.
                success_prev = success
                success = suction_contact.evaluate_forces(
                    force_location, force*i, external_moment, vac_level, obj_model, a_v)
            #force의 배율을 조정해가면서 이 이상이면 성공, 이하이면 실패인 배율을 찾아서 결과에 해당 배율로 곱해진 것을 추가
            #즉, i*force의 크기(=i)가 너무 커지면 너무 큰 외력(과 그에 의해 발생하는 돌림힘)으로 인해 evaluate_force 함수가 실패를 반환할 것이며, 이 한계 값을 찾는 과정.
            #i*force는 결국 물체에 작용하는 외력(중력, 충격 등)에 해당하는 물리적 의미를 가지며, 이 i값은 해당 force 방향으로 외력이 가해졌을 때 밀봉이 유지될 수 있는 가장 큰 힘의 크기를 나타낸다.
            #또한, i*force는 grasp wrench hull을 이루는 개별의 wrench가 되기도 한다.
            results.append(i*force)

    elif method == "binary_search":
        #위 코드에서 수정 : 상수만큼 이동시키는 것이 아니라 이분탐색으로 밀봉이 유지되는 최대의 힘을 찾는다.
        #비교 결과 위 코드와 시간 차이가 그렇게 유의미하게 나지는 않음.
        #나중에 MAX_FORCE의 값이 80이 아니라 한 300정도까지 늘어나야 유의미하게 차이가 날 것임.
        #print("이분 탐색")
        max_force = 80
        for force in force_directions:
            success_zero = suction_contact.evaluate_forces(
                force_location, force*0, external_moment, vac_level, obj_model, a_v)
            if success_zero == False:
                return 0.
            success_max = suction_contact.evaluate_forces(
                force_location, force*max_force, external_moment, vac_level, obj_model, a_v)
            if success_max == True:
                print(success_max, "KAY SE DOGAJA")
                return 0.
            left = 0
            right = max_force
            for _ in range(6):
                mid = (left + right)/2
                success_mid = suction_contact.evaluate_forces(
                    force_location, force*mid, external_moment, vac_level, obj_model, a_v)
                if success_mid:
                    left = mid
                else:
                    right = mid
            results.append(left*force)


    #결과 값은 해당 결과값들의 볼록 껍질(wrench hull)
    convex_hull = ss.ConvexHull(np.array(results))

    #그 볼록 껍질의 부피(단위 : N^3)를 반환.
    return convex_hull.volume


def contact_test_seal(con_p: np.ndarray,
                      obj_model: sclib.ModelData,
                      a_v: Optional[np.ndarray] = None,
                      noise_samples: int = 5,
                      noise_cone_angle: float = 0.1,
                      ) -> Tuple[sclib.SuctionContact, float, Union[np.ndarray, None]]:
    """ Tests whether the given contact point can form a seal or not.

    Parameters
    ----------
    con_p : np.ndarray
        _description_
    obj_model : sclib.ModelData
        _description_
    a_v : Optional[np.ndarray], optional
        _description_, by default None
    noise_samples : int, optional
        _description_, by default 5
    noise_cone_angle : float, optional
        _description_, by default 0.1

    Returns
    -------
    Tuple[sclib.SuctionContact, float, Union[np.ndarray, None]]
        - Returned suction contact object
        - The seal score for the given contact: -1 in case of failure; [0,1] valid seal score.
        - The approach vector for which the contact was tested. None if there is an invalid contact.
    """
    
    #지점 con_p에 대해 밀봉 형성 여부 검사 클래스 생성
    suction_contact = sclib.SuctionContact(con_p)
    # Form seal
    suction_contact.form_seal(obj_model)
    if suction_contact.success == False: #밀봉 실패 시
        return suction_contact, -1, None
    else: #밀봉 성공 시
        mean_point = np.mean(suction_contact.perimiter, axis = 0)
        delta_dist = np.abs(np.linalg.norm(mean_point - suction_contact.p_0))
        if delta_dist > 13: #형성된 perimeter의 평균 지점이랑 p_0이랑 거리가 13mm 초과로 떨어지면 suction score 0?
            return suction_contact, 0, -suction_contact.average_normal

    #a_v의 방향에 약간의 노이즈를 주어 랜덤성 부여
    if noise_samples == 0: #노이즈 샘플 없을 시
        #a_v의 값을 입력하지 않으면 일단 n_avg의 반대 방향 벡터로 설정
        if a_v is None:
            a_v = -suction_contact.average_normal
        #일단 a_v 방향으로 테스트
        suction_contact, contact_success = _test_for_seal(
            suction_contact, a_v, obj_model)
        if contact_success:
            return suction_contact, 1., a_v
        else:
            return suction_contact, 0., a_v
    else: #노이즈 샘플 입력 시
        if a_v is None:
            a_v = scf.vector_with_noise(-suction_contact.average_normal, noise_cone_angle)

        success_count = 0
        for i in range(noise_samples):
            a_v_noise = scf.vector_with_noise(a_v, noise_cone_angle)
            suction_contact, contact_success = _test_for_seal(
                suction_contact, a_v_noise, obj_model)
            if contact_success:
                success_count += 1
        #노이즈 적용에 따른 성공률이 score가 됨.
        return suction_contact, success_count/noise_samples, a_v

def _test_for_seal(suction_contact: sclib.SuctionContact, a_v, obj_model):
    # Analyze the seal and return the results
    if suction_contact.success == True:
        contact_success = suction_contact.evaluate_contact(a_v, obj_model)
        return suction_contact, contact_success
    else:
        return suction_contact, False
