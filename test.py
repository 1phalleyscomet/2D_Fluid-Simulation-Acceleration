def diffuse_velocity(self, iterations=20):
        u0, v0 = self.u.copy(), self.v.copy()
        a = self.dt * self.visc * (self.res - 2)**2
        for _ in range(iterations):
            self.u[1:-1,1:-1] = (u0[1:-1,1:-1] + a * (self.u[2:,1:-1] + self.u[:-2,1:-1] +
                                                        self.u[1:-1,2:] + self.u[1:-1,:-2])) / (1 + 4*a)
            self.v[1:-1,1:-1] = (v0[1:-1,1:-1] + a * (self.v[2:,1:-1] + self.v[:-2,1:-1] +
                                                        self.v[1:-1,2:] + self.v[1:-1,:-2])) / (1 + 4*a)

    def project(self, iterations=20):
        res = self.res
        h = 1.0 / (res - 2)
        div = np.zeros((res, res))
        p = np.zeros((res, res))
        div[1:-1,1:-1] = -0.5 * h * (self.u[2:,1:-1] - self.u[:-2,1:-1] +
                                        self.v[1:-1,2:] - self.v[1:-1,:-2])
        for _ in range(iterations):
            p[1:-1,1:-1] = (div[1:-1,1:-1] + p[2:,1:-1] + p[:-2,1:-1] +
                            p[1:-1,2:] + p[1:-1,:-2]) / 4
        self.u[1:-1,1:-1] -= 0.5 * (p[2:,1:-1] - p[:-2,1:-1]) / h
        self.v[1:-1,1:-1] -= 0.5 * (p[1:-1,2:] - p[1:-1,:-2]) / h

    def get_velocity(self):
        return self.u, self.v