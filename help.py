'''
utf-8;
author:Jinrong_Wu;
data:2022-07-10
We should use physics unit rad, m, (-pi,pi], theta_4 \ne 0.
For a 6R robot, thetas are only variables.
'''
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pi = torch.pi
class Trans_matrix():
    def __init__(self,alpha_last,a_last,d,theta):
        self.alpha = alpha_last.to(device)
        self.a  = a_last.to(device)
        self.d = d.to(device)
        self.theta = theta.to(device)
    
    def screw_matrix_x(self):
        alpha_x = self.alpha
        a_x = self.a
        screw_matrix_1 = torch.cos(alpha_x)
        screw_matrix_2 = torch.sin(alpha_x)
        screw_matrix = torch.tensor([[1,0,0,a_x],[0,screw_matrix_1,-screw_matrix_2,0],[0,screw_matrix_2,screw_matrix_1,0],[0,0,0,1]],device = device)
        # screw_matrix[abs(screw_matrix) <= 5e-8]=0
        return screw_matrix
    
    def screw_matrix_z(self):
        alpha_z = self.theta
        a_z = self.d
        screw_matrix_1 = torch.cos(alpha_z)
        screw_matrix_2 = torch.sin(alpha_z)
        screw_matrix = torch.tensor([[screw_matrix_1,-screw_matrix_2,0,0],[screw_matrix_2,screw_matrix_1,0,0],[0,0,1,a_z],[0,0,0,1]],device = device)
        # screw_matrix[abs(screw_matrix) <= 5e-8]=0
        return screw_matrix
    
    def trans_matrix(self):
        trans_matrix  = torch.mm(self.screw_matrix_x(),self.screw_matrix_z())
        return trans_matrix

    

class Series_forward():
    def __init__(self,T_0,T_1,T_2,T_3,T_4,T_5):
        self.T_0 = T_0.to(device)
        self.T_1 = T_1.to(device)
        self.T_2 = T_2.to(device)
        self.T_3 = T_3.to(device) 
        self.T_4 = T_4.to(device)
        self.T_5 = T_5.to(device)
    def forward(self):
        T_0_matrix = Trans_matrix(self.T_0[0],self.T_0[1],self.T_0[2],self.T_0[3]).trans_matrix()
        T_1_matrix = Trans_matrix(self.T_1[0],self.T_1[1],self.T_1[2],self.T_1[3]).trans_matrix()
        T_2_matrix = Trans_matrix(self.T_2[0],self.T_2[1],self.T_2[2],self.T_2[3]).trans_matrix()
        T_3_matrix = Trans_matrix(self.T_3[0],self.T_3[1],self.T_3[2],self.T_3[3]).trans_matrix()
        T_4_matrix = Trans_matrix(self.T_4[0],self.T_4[1],self.T_4[2],self.T_4[3]).trans_matrix()
        T_5_matrix = Trans_matrix(self.T_5[0],self.T_5[1],self.T_5[2],self.T_5[3]).trans_matrix()
        T_ori = T_0_matrix@T_1_matrix@T_2_matrix@T_3_matrix@T_4_matrix@T_5_matrix
        return T_ori
    
    
    
    


'''
The 0-2 axis have four solutions.
'''
class Solver():
    def __init__(self,T_ori,T_0,T_1,T_2,T_3,T_4,T_5):
        '''
        Given T_ori, we know T_ori = T_0*T_1*T_2*T_3*T_4*T_5, to solve the thetas. 
        @  = torch.mm()
        T_ori is a 4*4 matrix, and T_* are 1*3 vertor.
        '''
        self.T_ori = T_ori.to(device)
        self.T_0 = T_0.to(device)
        self.T_1 = T_1.to(device)
        self.T_2 = T_2.to(device)
        self.T_3 = T_3.to(device) 
        self.T_4 = T_4.to(device)
        self.T_5 = T_5.to(device)
    
    
    def judge_theta(self,theta):
        theta[theta<=-pi]+=2*pi
        theta[theta>pi] -=2*pi
        return theta
        
    def sol_theta_0(self):
        d_3 = self.T_2[2]
        p_x, p_y = self.T_ori[0,3], self.T_ori[1,3]
        theta_0_0 = torch.atan2(p_y,p_x) - torch.atan2(d_3,torch.sqrt(p_x**2+p_y**2-d_3**2))
        theta_0_1 = torch.atan2(p_y,p_x) - torch.atan2(d_3,-torch.sqrt(p_x**2+p_y**2-d_3**2))
        self.theta_0 = torch.tensor([theta_0_0,theta_0_1],device = device)
        return self.judge_theta(self.theta_0)

    def sol_theta_1(self,theta_0,theta_2):
        s_1 = torch.sin(theta_0)
        c_1 = torch.cos(theta_0)
        s_3 = torch.sin(theta_2)
        c_3 = torch.cos(theta_2)
        p_x, p_y, p_z = self.T_ori[0,3], self.T_ori[1,3], self.T_ori[2,3]
        a_2,d_3 = self.T_2[1], self.T_2[2]
        a_3,d_4 = self.T_3[1], self.T_3[2] 
        theta_23_y = -(a_3+a_2*c_3)*p_z-(c_1*p_x+s_1*p_y)*(d_4-a_2*s_3)
        theta_23_x = (a_2*s_3-d_4)*p_z+(a_3+a_2*c_3)*(c_1*p_x+s_1*p_y)
        theta_23 = torch.atan2(theta_23_y,theta_23_x)
        theta2 = theta_23-theta_2
        self.theta_1 = theta2
        return self.judge_theta(self.theta_1)
#     def sol_theta_1(self,theta_1,theta_3):
#         s_1 = torch.sin(theta_1)
#         c_1 = torch.cos(theta_1)
#         c_3 = torch.sin(theta_3)
#         s_3 = torch.cos(theta_3)
#         p_x = self.T_ori[0,3]
#         p_y = self.T_ori[1,3]
#         p_z = self.T_ori[2,3]
#         a_2,d_3 = self.T_2[1],self.T_2[2]
#         a_3,d_4 = self.T_3[1],self.T_3[2]
#         theta_23_y = (-a_3-a_2*c_3)*p_z - (c_1*p_x+s_1*p_y)*(d_4-a_2*s_3)
#         theta_23_x = (a_2*s_3-d_4)*p_z+(a_3+a_2*c_3)*(c_1*p_x+s_1*p_y)
# #        《机器人学导论》P83 Eq.(4.72),error
#         theta23 = torch.atan2(theta_23_y,theta_23_x)
#         theta2 = theta23-theta_3
#         self.theta_1 = theta2
#         return self.theta_1
        
    def sol_theta_2(self):
        a_3 = self.T_3[1]
        d_4 = self.T_3[2]
        K = self.T_ori[0,3]**2+self.T_ori[1,3]**2+self.T_ori[2,3]**2 - self.T_2[1]**2-self.T_3[1]**2-self.T_2[2]**2-self.T_3[2]**2
        K = K/(2*self.T_2[1])
        theta_2_0 = torch.atan2(a_3,d_4) - torch.atan2(K,torch.sqrt(a_3**2+d_4**2-K**2))
        theta_2_1 = torch.atan2(a_3,d_4) - torch.atan2(K,-torch.sqrt(a_3**2+d_4**2-K**2))
        self.theta_2 = torch.tensor([theta_2_0,theta_2_1],device = device)
        # print(self.theta_2)
        return self.judge_theta(self.theta_2)    
    


    def sol_theta_3(self,theta_0,theta_1,theta_2):
        r_13 = self.T_ori[0,2]
        r_23 = self.T_ori[1,2]
        r_33 = self.T_ori[2,2]
        theta_23 = theta_1 + theta_2
        c_23 = torch.cos(theta_23)
        s_23 = torch.sin(theta_23)
        s_1 = torch.sin(theta_0)
        c_1 = torch.cos(theta_0)
        p_y = -r_13*s_1+r_23*c_1
        p_x = -r_13*c_1*c_23-r_23*s_1*c_23+r_33*s_23
        if abs(p_y-0)<=1e-7 and abs(p_x-0)<=1e-7:
            print('Singular position: theta_5=0, and theta_4 $\tehta_4$ is free in (-pi,pi], depends on theta_6 $\tehta_6$')
        else:
            theta_4 = torch.atan2(p_y,p_x)
            self.theta_3 = theta_4
        return self.judge_theta(self.theta_3)
    
    def sol_theta_4(self,theta_0,theta_1,theta_2,theta_3):
        r_13 = self.T_ori[0,2]
        r_23 = self.T_ori[1,2]
        r_33 = self.T_ori[2,2]
        theta_23 = theta_1 + theta_2
        c_23 = torch.cos(theta_23)
        s_23 = torch.sin(theta_23)
        s_1 = torch.sin(theta_0)
        c_1 = torch.cos(theta_0)
        s_4 = torch.sin(theta_3)
        c_4 = torch.cos(theta_3)
        p_y = -r_13*(c_1*c_23*c_4+s_1*s_4)-r_23*(s_1*c_23*c_4-c_1*s_4)+r_33*(s_23*c_4)
        p_x = -r_13*c_1*s_23-r_23*s_1*s_23-r_33*c_23
        theta_5 = torch.atan2(p_y,p_x)
        self.theta_4 = theta_5
        return self.judge_theta(self.theta_4)
        
    def sol_theta_5(self,theta_0,theta_1,theta_2,theta_3,theta_4):
        r_11 = self.T_ori[0,0]
        r_21 = self.T_ori[1,0]
        r_31 = self.T_ori[2,0]
        theta_23 = theta_1 + theta_2
        c_23 = torch.cos(theta_23)
        s_23 = torch.sin(theta_23)
        s_1 = torch.sin(theta_0)
        c_1 = torch.cos(theta_0)
        s_4 = torch.sin(theta_3)
        c_4 = torch.cos(theta_3)
        s_5 = torch.sin(theta_4)
        c_5 = torch.cos(theta_4)
        p_y = -r_11*(c_1*c_23*s_4-s_1*c_4)-r_21*(s_1*c_23*s_4+c_1*c_4)+r_31*s_23*s_4
        p_x = r_11*((c_1*c_23*c_4+s_1*s_4)*c_5-c_1*s_23*s_5)+r_21*((s_1*c_23*c_4-c_1*s_4)*c_5-s_1*s_23*s_5)-r_31*(s_23*c_4*c_5+c_23*s_5)
        theta_6 = torch.atan2(p_y,p_x)
        self.theta_5 = theta_6
        return self.judge_theta(self.theta_5)
    
        

    def solver(self):
        solutions_array = torch.empty((8,6),device = device)
        theta_0_array = self.sol_theta_0() 
        theta_2_array = self.sol_theta_2()
        theta_1_array = torch.empty(4,device = device)
        theta_02_cart_prod = torch.cartesian_prod(theta_0_array, theta_2_array)
        for i in torch.arange(theta_02_cart_prod.shape[0]):
            theta_1_array[i] = self.sol_theta_1(theta_02_cart_prod[i,0],theta_02_cart_prod[i,1])
        solutions_array[:4,0] = theta_02_cart_prod[:4,0]
        solutions_array[:4,1] = theta_1_array
        solutions_array[:4,2] = theta_02_cart_prod[:4,1]
        solutions_array[4:8,:] = solutions_array[:4,:]
        for i in range(4):
            theta_3 = self.sol_theta_3(solutions_array[i,0],solutions_array[i,1],solutions_array[i,2])
            theta_4 = self.sol_theta_4(solutions_array[i,0],solutions_array[i,1],solutions_array[i,2],theta_3)
            theta_5 = self.sol_theta_5(solutions_array[i,0],solutions_array[i,1],solutions_array[i,2],theta_3,theta_4)
            theta_345_array_0 = torch.tensor([theta_3,theta_4,theta_5],device = device)
            theta_345_array_1 = self.judge_theta(torch.tensor([theta_3+pi,-theta_4,theta_5+pi],device = device))
            solutions_array[i,3:6] = theta_345_array_0
            solutions_array[i+4,3:6] = theta_345_array_1
        return   solutions_array   
            
        
