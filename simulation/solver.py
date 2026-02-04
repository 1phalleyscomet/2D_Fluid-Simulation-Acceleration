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
        self.u.copy()->timestep마다 메모리 재할당으로 속도 저하
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
    def diffuse(self, iterations=20):
        #현 속도장 백업(이전 값 저장)
        self.u0[:] = self.u
        self.v0[:] = self.v
        #확산 계수
        a = self.dt * self.visc * (self.res - 2)**2
        #jacobi 반복 (interations 만큼)
        for _ in range(iterations):
            self.u[1:-1,1:-1] = (self.u0[1:-1,1:-1] + a * (self.u[2:,1:-1] + self.u[:-2,1:-1] + self.u[1:-1,2:] + self.u[1:-1,:-2])) / (1 + 4*a)
            self.v[1:-1,1:-1] = (self.v0[1:-1,1:-1] + a * (self.v[2:,1:-1] + self.v[:-2,1:-1] + self.v[1:-1,2:] + self.v[1:-1,:-2])) / (1 + 4*a)


    #gradient 제거
    def project(self, iterations=20):
        #기본 설정
        res = self.res #격자 크기
        h = 1.0 / (res - 2) #격자단위
        div = np.zeros((res, res)) #divergence 저장
        p = np.zeros((res, res)) #pressure 저장
        div[1:-1,1:-1] = -0.5 * h * (self.u[2:,1:-1] - self.u[:-2,1:-1] +
                                        self.v[1:-1,2:] - self.v[1:-1,:-2])
        #jacobi 반복(pressure cal.)
        for _ in range(iterations):
            p[1:-1,1:-1] = (div[1:-1,1:-1] + p[2:,1:-1] + p[:-2,1:-1] +
                            p[1:-1,2:] + p[1:-1,:-2]) / 4
        #속도장 보정(속도장 div 제거)
        self.u[1:-1,1:-1] -= 0.5 * (p[2:,1:-1] - p[:-2,1:-1]) / h
        self.v[1:-1,1:-1] -= 0.5 * (p[1:-1,2:] - p[1:-1,:-2]) / h
        """
        비압축 유체를 가정 ->div=0으로 속도 발산 제거
        속도에서 압력 기울기(pressure gradient) 제거(유체 압축 X)
        """


    def get_data(self):
        #ai 학습용으로 속도장 stack 반환
        return np.stack([self.u,self.v],axis=0).astype(np.float32)