import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import time

class Stable_Fluid:
    def __init__(self,res,dt,visc):
        #격자 및 파라미터 초기화
        self.res=res #격자 크기
        self.dt=dt #시간 간격
        self.visc=visc #점성계수

        #속도장 초기화
        self.u=np.zeros((res,res))
        self.v=np.zeros((res,res))
        #이전 좌표 초기화
        self.u0 = np.zeros((res, res)) 
        self.v0 = np.zeros((res, res))

    def add_force(self, x,y,fx,fy):
        #특정 위치에서 속도(힘) 추가
        self.u[x,y] +=fx
        self.v[x,y] +=fy

    def step(self): 
        #현재 속도를 이전 속도로 복사
        self.u0[:] = self.u
        self.v0[:] = self.v
        """
        근데 이러면 self.u0 = self.u.copy()랑 무슨 차이인지??
        ->timestep마다 메모리 재할당으로 속도 저하
        u0가 새로운 객체를 point함
        """

        #물리법칙 timestep
        self.diffuse()
        #self.project()
        self.advect()
        self.project()
        
    #이류
    def advect(self):
        res=self.res
        dt0=self.dt*(res-2)

        #격자 좌표 구간 생성
        X, Y = np.meshgrid(np.arange(res), np.arange(res))

        #backtrace(semi-ragrangian)
        x_back = X - dt0 * self.u0
        y_back = Y - dt0 * self.v0

        # 경계 처리
        x_back = np.clip(x_back, 0.5, res-1.5)
        y_back = np.clip(y_back, 0.5, res-1.5)

        # map_coordinates 보간
        self.u = map_coordinates(self.u0, [y_back, x_back], order=1, mode='nearest')
        self.v = map_coordinates(self.v0, [y_back, x_back], order=1, mode='nearest')
        #order=1->linear(선형 보간)
        #mode='nearest'->배열 밖 좌표를 가장 가까운 내부 값으로 대체

    #확산
    def diffuse(self):
        pass

    #gradient 제거
    def project(self):
        pass

    def get_data(self):
        #ai 학습용으로 속도장 stack 반환
        return np.stack([self.u,self.v],axis=0)