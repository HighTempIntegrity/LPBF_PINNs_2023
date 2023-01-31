from ImportFile import *
from EquationBaseClass import EquationBaseClass
from GeneratorPoints import generator_points
from SquareDomain import SquareDomain
from BoundaryConditions import PeriodicBC, DirichletBC, AbsorbingBC, NeumannBC

class EquationClass(EquationBaseClass):

    def __init__(self, ):
        EquationBaseClass.__init__(self)

        self.type_of_points = "will be defined from outside"
        self.output_dimension = 1
        self.space_dimensions = 3
        self.time_dimensions = 1


        # self.parameter_dimensions = 4
        self.parameters_values = torch.tensor([[0.005, 0.02],  # conductivity/1000
                                               [0.002, 0.005],  # density in kg/mm3 *Cp
                                               [0.005, 0.02],  # Laser Power/1000
                                               [0.5, 1.5],      # laser speed mm/s/1000
                                               [0.05, 0.1],    # Goldak a [mm]
                                               [0.05, 0.1],    # Goldak b [mm]
                                               [0.05, 0.1],    # Goldak cf [mm]
                                               [0.2, 0.3]])   # Goldak cr [mm]                        
        
        self.parameter_dimensions = self.parameters_values.size(dim=0)

   
        self.extrema_values = torch.tensor([[-0.1, 1],
                                            [-1, 1.8],
                                            [-1, 1],
                                            [-1, 0.03]])
        self.vm=1.5 #distance the laser travels for t_input=1
        self.tmax=0.0015 #not needed with speed as input parameter
        self.umax = 1600
        self.list_of_BC = list([[self.ub1, self.ub1], [self.ub1, self.ub1], [self.ub1, self.ub1]])
        self.extrema_values = self.extrema_values if self.parameters_values is None else torch.cat(
            [self.extrema_values, self.parameters_values], 0)
        self.square_domain = SquareDomain(self.output_dimension,
                                          self.time_dimensions,
                                          self.space_dimensions,
                                          self.list_of_BC,
                                          self.extrema_values,
                                          self.type_of_points,
                                          vel_wave=self.a)
        self.setuparr=np.array([]) #which setups should be used for support?

        #self.heat = torch.full(size=(2**19,5), fill_value=float('nan'))
   

    def add_collocation_points(self, n_coll, random_seed):
        self.square_domain.type_of_points = self.type_of_points
        return self.square_domain.add_collocation_points(n_coll, random_seed)

    def add_boundary_points(self, n_boundary, random_seed):
        return self.square_domain.add_boundary_points(n_boundary, random_seed)

    def add_initial_points(self, n_initial, random_seed):
        extrema_0 = self.extrema_values[:, 0]  # minimum values
        extrema_f = self.extrema_values[:, 1]  # maximum values
        x_time_0 = generator_points(n_initial, self.time_dimensions + self.space_dimensions + self.parameter_dimensions,
                                    random_seed, "initial_center", True)
        x_time_0[:, 0] = torch.full(size=(n_initial,), fill_value=0.0)
        x_time_0 = x_time_0 * (extrema_f - extrema_0) + extrema_0

        y_time_0 = self.u0(x_time_0)

        return x_time_0, y_time_0

    def a(self, x):
        return 1

    def apply_bc(self, model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list):

        self.square_domain.apply_boundary_conditions(model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list)
        print('Applied BC')

        folder = 'Points'

        

    def apply_ic(self, model, x_u_train, u_train, u_pred_var_list, u_train_var_list):
        for j in range(self.output_dimension):
            if x_u_train.shape[0] != 0:
                u = model(x_u_train)[:, j]
                u_pred_var_list.append(u)
                u_train_var_list.append(u_train[:])
                L2=torch.mean((u - u_train[:]) ** 2) / torch.mean((u_train[:]) ** 2)
                max=torch.max(u-u_train[:])
                print('init+support relative L2',L2.detach().cpu().numpy().round(4))
                print('init+support max difference',max.detach().cpu().numpy().round(4))

    def compute_res(self, network, x_f_train, solid_object):
        self.network = network
        x_f_train.requires_grad = True
        u = network(x_f_train)[:, 0].reshape(-1, )
        grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones_like(u).to(self.device), create_graph=True)[
            0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_y = grad_u[:, 2]
        grad_u_z = grad_u[:, 3]

        rhocp = x_f_train[:, 5]
        c = x_f_train[:, 4]

        grad_u_xx = torch.autograd.grad(c * grad_u_x, x_f_train, grad_outputs=torch.ones_like(u).to(self.device),
                                        create_graph=True)[0][:, 1]
        grad_u_yy = torch.autograd.grad(c * grad_u_y, x_f_train, grad_outputs=torch.ones_like(u).to(self.device),
                                        create_graph=True)[0][:, 2]
        grad_u_zz = torch.autograd.grad(c * grad_u_z, x_f_train, grad_outputs=torch.ones_like(u).to(self.device),
                                        create_graph=True)[0][:, 3]
        time = self.vm/(x_f_train[:, 7]*1000)
        # time = 1 / 1000
        q = self.goldak_source(x_f_train) * time / (rhocp * self.umax)


        res2 = (grad_u_xx.reshape(-1, ) + grad_u_yy.reshape(-1, ) + grad_u_zz.reshape(-1, )) * time / (rhocp)
        residual = (grad_u_t.reshape(-1, ) - res2 - q)

        # enforce zero-gradient in the region before laser gradient
        mask_init = torch.le(x_f_train[:, 0], 0)
        residual[mask_init] = grad_u_t.reshape(-1, )[mask_init]
        # enforce temperatures above 0
        value=25
        mask_temp = torch.le(u*self.umax, value)
        residual[mask_temp] = abs(u[mask_temp]*self.umax-value)*residual[mask_temp]/abs(residual[mask_temp]) + residual[mask_temp]


        #debugging printout
        print('--Residual--')
        print("Max du/dt", torch.max(abs(grad_u_t.reshape(-1, ))).detach().cpu().numpy().round(4), "; mean: ",
              torch.mean(grad_u_t.reshape(-1, )).detach().cpu().numpy().round(4))
        print("Max d/dx(c*du/dx)", torch.max(abs(res2)).detach().cpu().numpy().round(4), "; mean: ",
              torch.mean(res2).detach().cpu().numpy().round(4))
        print("Max source", (torch.max(abs(q))).detach().cpu().numpy().round(4), "; mean: ",
              (torch.mean(q)).detach().cpu().numpy().round(4), "; Min source: ", (torch.min(q)).detach().cpu().numpy().round(4))
        print("max predicted temp: ", (torch.max((u * self.umax))).detach().cpu().numpy().round(4), "min temp: ",
              (torch.min((u * self.umax))).detach().cpu().numpy().round(4))
        print("Max residual: ", torch.max(abs(residual)).detach().cpu().numpy().round(4), "; Mean: ",
              torch.mean(residual).detach().cpu().numpy().round(4))

        return residual

    def v0(self, x):
        return torch.zeros((x.shape[0], 1))

    def ub0(self, t):
        type_BC = [DirichletBC()]
        u0 = 25 / self.umax
        u = torch.full(size=(t.shape[0], 1), fill_value=u0)
        return u, type_BC

    def ub1(self, t):
        type_BC = [NeumannBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def ub2(self, t):
        type_BC = [NeumannBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def u0(self, x):
        self.ini = 25 / self.umax
        u0 = torch.ones(x.shape[0]) * self.ini
        return u0

    def source(self, x):
        sig = 0.05# x[:,8]
        q = x[:,6]*1000
        n = 1
        x_phys = x[:, 1]
        y_phys = x[:, 2]
        z_phys = x[:, 3]
        v = self.vm
        #v=x[:,7]*1000*self.tmax
        timestep = -0. + x[:, 0] * v
        mask = 1 - 1. / (1 + torch.exp((x[:, 0] - 0.03) * 400)) #have a logistic function as ramp in the beginning at t=0
        heat = 6 * np.sqrt(3) * n * q / (pi * np.sqrt(pi) * sig ** 3) * torch.exp(
            -3 * ((x_phys - timestep) ** 2 + (y_phys) ** 2 + (z_phys - 0.03) ** 2) / sig ** 2) * mask

        # heat=heat/torch.max(heat)
        return heat

    def goldak_source(self, input):
        q = input[:, 6] * 1000     # laser power input [W/1000]

        x = input[:, 1]              # spatial x-coordinate [mm]
        y = input[:, 2]              # spatial y-coordinate [mm]
        z = input[:, 3]              # spatial z-coordinate [mm]


        print('--Heat source coordinates--')
        print('Max x: ', torch.max(x).item(), ' | Min x: ', torch.min(x).item())
        print('Max y: ', torch.max(y).item(), ' | Min y: ', torch.min(y).item())
        print('Max z: ', torch.max(z).item(), ' | Min z: ', torch.min(z).item())


        a = input[:, 8]              # Goldak parameter a [mm]
        b = input[:, 9]              # Goldak parameter b [mm]
        cf = input[:, 10]             # Goldak parameter cf [mm]
        cr = input[:, 11]             # Goldak parameter cr [mm]

        v = self.vm
        timestep = input[:, 0] * v # - 0.01
        #timestep = input[:,0] * input[:,7]
        # logistic function as ramp at t=0
        mask = 1 - 1. / (1 + torch.exp((input[:, 0] - 0.03) * 400))

        ff = 2 / (1 + cr / cf)
        fr = 2 / (1 + cf / cr)

        argfact = 1000

        c = cf * (torch.full(q.size(), 0.5).to(self.device) + 0.5 * torch.tanh(argfact*(x-timestep)).to(self.device)) + \
            cr * (torch.full(q.size(), 0.5).to(self.device) + 0.5 * torch.tanh(argfact*(timestep-x)).to(self.device))

        f = ff * (torch.full(q.size(), 0.5).to(self.device) + 0.5 * torch.tanh(argfact*(x-timestep)).to(self.device)) + \
            fr * (torch.full(q.size(), 0.5).to(self.device) + 0.5 * torch.tanh(argfact*(timestep-x)).to(self.device))

        c.to(self.device)
        f.to(self.device)

        heat = (2 * f * q / (a * b * c) * (np.sqrt(3 / np.pi))**3 *
                torch.exp(-3 * ((x-timestep)**2 / c**2 + y**2 / b**2 + (z-0.03)**2 / a**2)) * mask)

        return heat


   

    