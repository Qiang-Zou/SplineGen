import torch

class CurveEval(torch.nn.Module):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __init__(self, m, dimension=3, p=3, out_dim=128, method='tc', dvc='cuda'):
        super(CurveEval, self).__init__()
        self.m = m
        self._dimension = dimension
        self.p=p
        self.u = torch.linspace(1e-5, 1.0-1e-5, steps=out_dim, dtype=torch.float32,device=dvc)
        self.method = method
        self.dvc = dvc

    def forward(self,input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # input will be of dimension (batch_size, m+1, n+1, dimension)
        ctrl_pts, knot_u= input

        U=knot_u
        # U_c = torch.cumsum(torch.where(knot_u<0.0, knot_u*0+1e-4, knot_u), dim=1)
        # U = (U_c - U_c[:,0].unsqueeze(-1)) / (U_c[:,-1].unsqueeze(-1) - U_c[:,0].unsqueeze(-1))

        if torch.isnan(U).any():
            # print(U_c)
            print(knot_u)

        #############################################################################
        #################### Gaussian gradient smoothening ##########################

        # u = self.u.unsqueeze(0)
        # v = self.v.unsqueeze(0)

        # uspan_uv = torch.stack([torch.min(torch.where((u - U[s,self.p:-self.p].unsqueeze(1))>1e-8, u - U[s,self.p:-self.p].unsqueeze(1), (u - U[s,self.p:-self.p].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+self.p for s in range(U.size(0))])
        # vspan_uv = torch.stack([torch.min(torch.where((v - V[s,self.q:-self.q].unsqueeze(1))>1e-8, v - V[s,self.q:-self.q].unsqueeze(1), (v - V[s,self.q:-self.q].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+self.q for s in range(V.size(0))])

        # Nu_uv = BasisFunc.apply(u, U, uspan_uv, self.p)

        # Nu_uv = Nu_uv.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)


        # Nv_uv = BasisFunc.apply(v, V, vspan_uv, self.q)

        # Nv_uv = Nv_uv.unsqueeze(1).unsqueeze(-1).unsqueeze(-3)


        #############################################################################
        #################### Autograd based definition ##############################
        u = self.u.unsqueeze(0)
        uspan_uv = torch.stack([torch.min(torch.where((u - U[s,self.p:-self.p].unsqueeze(1))>1e-8, u - U[s,self.p:-self.p].unsqueeze(1), (u - U[s,self.p:-self.p].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+self.p for s in range(U.size(0))])

        u = u.squeeze(0)
        Ni = [u*0 for i in range(self.p+1)]
        Ni[0] = u*0 + 1
        for k in range(1,self.p+1):
            saved = (u)*0.0
            for r in range(k):
                UList1 = torch.stack([U[s,uspan_uv[s,:] + r + 1] for s in range(U.size(0))])
                UList2 = torch.stack([U[s,uspan_uv[s,:] + 1 - k + r] for s in range(U.size(0))])
                temp = Ni[r]/((UList1 - u) + (u - UList2))
                temp = torch.where(((UList1 - u) + (u - UList2))==0.0, u*0+1e-4, temp)
                Ni[r] = saved + (UList1 - u)*temp
                saved = (u - UList2)*temp
            Ni[k] = saved

        Nu_uv = torch.stack(Ni).permute(1,0,2).unsqueeze(-1)

        ################################################################################
        ################################################################################

        pts = torch.stack([torch.stack([ctrl_pts[s,(uspan_uv[s,:]-self.p+l),:] for l in range(self.p+1)]) for s in range(U.size(0))])


        # rational_pts = pts[:, :, :, :, :, :self._dimension]*pts[:, :, :, :, :, self._dimension:]
        # pts = torch.cat((rational_pts,pts[:, :, :, :, :, self._dimension:]),-1)

        # print((Nu_uv*Nv_uv).size(), pts.size())
        curve = torch.sum((Nu_uv*pts), dim=1)

        # surfaces = torch.sum((Nu_uv*pts), (1,2))
        # print(surfaces[:,:,:,self._dimension].sum())
        # # print(surfaces.size())
        # surfaces = surfaces[:,:,:,:self._dimension]#/surfaces[:,:,:,self._dimension].unsqueeze(-1)
        return curve

    def Max_MSE_Error(self,input,data_points:torch.Tensor):
        pred_points:torch.Tensor=self(input)

        diff=torch.linalg.norm(data_points-pred_points,dim=-1)

        max=torch.max(diff,dim=1)[0]
        mse=torch.mean(diff,dim=1)

        return max,mse

class CurveEval2(torch.nn.Module):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __init__(self, m, dimension=3, p=3, out_dim=128, method='tc', dvc='cuda'):
        super(CurveEval2, self).__init__()
        self.m = m
        self._dimension = dimension
        self.p=p
        self.u = torch.linspace(1e-5, 1.0-1e-5, steps=out_dim, dtype=torch.float32)
        self.method = method
        self.dvc = dvc

    def forward(self,input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # input will be of dimension (batch_size, m+1, n+1, dimension)
        ctrl_pts, knot_u= input

        U=knot_u
        # U_c = torch.cumsum(torch.where(knot_u<0.0, knot_u*0+1e-4, knot_u), dim=1)
        # U = (U_c - U_c[:,0].unsqueeze(-1)) / (U_c[:,-1].unsqueeze(-1) - U_c[:,0].unsqueeze(-1))

        if torch.isnan(U).any():
            # print(U_c)
            print(knot_u)

        #############################################################################
        #################### Gaussian gradient smoothening ##########################

        # u = self.u.unsqueeze(0)
        # v = self.v.unsqueeze(0)

        # uspan_uv = torch.stack([torch.min(torch.where((u - U[s,self.p:-self.p].unsqueeze(1))>1e-8, u - U[s,self.p:-self.p].unsqueeze(1), (u - U[s,self.p:-self.p].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+self.p for s in range(U.size(0))])
        # vspan_uv = torch.stack([torch.min(torch.where((v - V[s,self.q:-self.q].unsqueeze(1))>1e-8, v - V[s,self.q:-self.q].unsqueeze(1), (v - V[s,self.q:-self.q].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+self.q for s in range(V.size(0))])

        # Nu_uv = BasisFunc.apply(u, U, uspan_uv, self.p)

        # Nu_uv = Nu_uv.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)


        # Nv_uv = BasisFunc.apply(v, V, vspan_uv, self.q)

        # Nv_uv = Nv_uv.unsqueeze(1).unsqueeze(-1).unsqueeze(-3)


        #############################################################################
        #################### Autograd based definition ##############################
        u = self.u.unsqueeze(0)
        p=self.p
        # uspan_uv = torch.stack([torch.min(torch.where((u - U[s,self.p:-self.p].unsqueeze(1))>1e-8, u - U[s,self.p:-self.p].unsqueeze(1), (u - U[s,self.p:-self.p].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+self.p for s in range(U.size(0))])
        uspan_uv = torch.stack([torch.min(torch.where((u[s] - U[s,p:].unsqueeze(1))>1e-8, u[s] - U[s,p:].unsqueeze(1), (u[s] - U[s,p:].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+p for s in range(U.size(0))])

        # u = u.squeeze(0)
        # Ni = [u*0 for i in range(self.p+1)]
        # Ni[0] = u*0 + 1
        # for k in range(1,self.p+1):
        #     saved = (u)*0.0
        #     for r in range(k):
        #         UList1 = torch.stack([U[s,uspan_uv[s,:] + r + 1] for s in range(U.size(0))])
        #         UList2 = torch.stack([U[s,uspan_uv[s,:] + 1 - k + r] for s in range(U.size(0))])
        #         temp = Ni[r]/((UList1 - u) + (u - UList2))
        #         temp = torch.where(((UList1 - u) + (u - UList2))==0.0, u*0+1e-4, temp)
        #         Ni[r] = saved + (UList1 - u)*temp
        #         saved = (u - UList2)*temp
        #     Ni[k] = saved

        # Nu_uv = torch.stack(Ni).permute(1,0,2).unsqueeze(-1)
        u = self.u
        Ni = [u*0 for i in range(p+1)]
        Ni[0] = u*0 + 1
        for k in range(1,p+1):
            saved = (u)*0.0
            for r in range(k):
                UList1 = torch.stack([U[s,uspan_uv[s,:] + r + 1] for s in range(U.size(0))])
                UList2 = torch.stack([U[s,uspan_uv[s,:] + 1 - k + r] for s in range(U.size(0))])

                dU=((UList1 - u) + (u - UList2))
                dU_=torch.where(dU==0.0, u*0+1e-4, dU)

                temp = Ni[r]/dU_
                temp = torch.where(dU==0.0, u*0+1e-4, temp)
                Ni[r] = saved + (UList1 - u)*temp
                saved = (u - UList2)*temp
            Ni[k] = saved

        Nu_uv = torch.stack(Ni).permute(1,2,0)

        ################################################################################
        ################################################################################

        pts = torch.stack([torch.stack([ctrl_pts[s,(uspan_uv[s,:]-self.p+l),:] for l in range(self.p+1)]) for s in range(U.size(0))])


        # rational_pts = pts[:, :, :, :, :, :self._dimension]*pts[:, :, :, :, :, self._dimension:]
        # pts = torch.cat((rational_pts,pts[:, :, :, :, :, self._dimension:]),-1)

        # print((Nu_uv*Nv_uv).size(), pts.size())
        curve = torch.sum((Nu_uv*pts), dim=1)

        # surfaces = torch.sum((Nu_uv*pts), (1,2))
        # print(surfaces[:,:,:,self._dimension].sum())
        # # print(surfaces.size())
        # surfaces = surfaces[:,:,:,:self._dimension]#/surfaces[:,:,:,self._dimension].unsqueeze(-1)
        return curve

class CurveEval3(torch.nn.Module):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __init__(self, m, dimension=3, p=3, out_dim=128, method='tc', dvc='cuda'):
        super(CurveEval3, self).__init__()
        self.m = m
        self._dimension = dimension
        self.p=p
        self.u = torch.linspace(1e-5, 1.0-1e-5, steps=out_dim, dtype=torch.float32)
        self.method = method
        self.dvc = dvc

    def forward(self,input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # input will be of dimension (batch_size, m+1, n+1, dimension)
        ctrl_pts, knot_u= input
        p=self.p

        # params,points,points_mask,knot_u,knots_len= input

        params=self.u.unsqueeze(0)

        device=params.device

        U=knot_u
        # U_c = torch.cumsum(torch.where(knot_u<0.0, knot_u*0+1e-4, knot_u), dim=1)
        # U = (U_c - U_c[:,0].unsqueeze(-1)) / (U_c[:,-1].unsqueeze(-1) - U_c[:,0].unsqueeze(-1))

        if torch.isnan(U).any():
            # print(U_c)
            print(knot_u)

        u = params.unsqueeze(1)
        uspan_uv = torch.stack([torch.min(torch.where((u[s] - U[s,p:].unsqueeze(1))>1e-8, u[s] - U[s,p:].unsqueeze(1), (u[s] - U[s,p:].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+p for s in range(U.size(0))])

        # torch._assert(torch.all(uspan_uv<25),"uspan out of bound")
        # print('uspan_uv:',uspan_uv)
        # print((u[0] - U[0,p:].unsqueeze(1))[...,35])

        u = u.squeeze(1)
        Ni = [u*0 for i in range(p+1)]
        Ni[0] = u*0 + 1
        for k in range(1,p+1):
            saved = (u)*0.0
            for r in range(k):
                UList1 = torch.stack([U[s,uspan_uv[s,:] + r + 1] for s in range(U.size(0))])
                UList2 = torch.stack([U[s,uspan_uv[s,:] + 1 - k + r] for s in range(U.size(0))])

                dU=((UList1 - u) + (u - UList2))
                dU_=torch.where(dU==0.0, u*0+1e-4, dU)

                temp = Ni[r]/dU_
                temp = torch.where(dU==0.0, u*0+1e-4, temp)
                Ni[r] = saved + (UList1 - u)*temp
                saved = (u - UList2)*temp
            Ni[k] = saved

        Nu_uv = torch.stack(Ni).permute(1,2,0)
        # torch.linalg.lstsq()
        scatter_index=torch.arange(-p,1,device=device).unsqueeze(0).unsqueeze(0).expand(uspan_uv.shape[0],uspan_uv.shape[1],-1)
        scatter_index=uspan_uv.unsqueeze(-1)+scatter_index

        N_all=torch.zeros((Nu_uv.size(0),Nu_uv.size(1),ctrl_pts.shape[1]),device=device)
        # print(143,scatter_index.dtype,Nu_uv.dtype,N_all.dtype)
        N_all=N_all.scatter_(2,scatter_index,Nu_uv)
        # N_mask=points_mask.unsqueeze(-1).expand(-1,-1,30)

        # N_all=torch.masked_fill(N_all,~N_mask.bool(),0)
            # N_all[s,uspan_uv[s,:]]

        solution=ctrl_pts.to(dtype=torch.float32)
        results=N_all@solution


        return results

def getCtrlPts(p,input):
    # input will be of dimension (batch_size, m+1, n+1, dimension)
    # 1 batch
    params,points,points_mask,knot_u,knots_len= input
    device=params.device

    params=torch.masked_fill(params,~points_mask.bool(),0)
    points=torch.masked_fill(points,~(points_mask.unsqueeze(-1).expand(-1,-1,points.shape[-1])).bool(),0)

    U=knot_u
    # U_c = torch.cumsum(torch.where(knot_u<0.0, knot_u*0+1e-4, knot_u), dim=1)
    # U = (U_c - U_c[:,0].unsqueeze(-1)) / (U_c[:,-1].unsqueeze(-1) - U_c[:,0].unsqueeze(-1))

    if torch.isnan(U).any():
        # print(U_c)
        print(knot_u)

    u = params.unsqueeze(1)
    uspan_uv = torch.stack([torch.min(torch.where((u[s] - U[s,p:].unsqueeze(1))>1e-8, u[s] - U[s,p:].unsqueeze(1), (u[s] - U[s,p:].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+p for s in range(U.size(0))])

    # torch._assert(torch.all(uspan_uv<25),"uspan out of bound")
    # print('uspan_uv:',uspan_uv)
    # print((u[0] - U[0,p:].unsqueeze(1))[...,35])

    u = u.squeeze(1)
    Ni = [u*0 for i in range(p+1)]
    Ni[0] = u*0 + 1
    for k in range(1,p+1):
        saved = (u)*0.0
        for r in range(k):
            UList1 = torch.stack([U[s,uspan_uv[s,:] + r + 1] for s in range(U.size(0))])
            UList2 = torch.stack([U[s,uspan_uv[s,:] + 1 - k + r] for s in range(U.size(0))])

            dU=((UList1 - u) + (u - UList2))
            dU_=torch.where(dU==0.0, u*0+1e-4, dU)

            temp = Ni[r]/dU_
            temp = torch.where(dU==0.0, u*0+1e-4, temp)
            Ni[r] = saved + (UList1 - u)*temp
            saved = (u - UList2)*temp
        Ni[k] = saved

    Nu_uv = torch.stack(Ni).permute(1,2,0)
    # torch.linalg.lstsq()
    scatter_index=torch.arange(-p,1,device=device).unsqueeze(0).unsqueeze(0).expand(uspan_uv.shape[0],uspan_uv.shape[1],-1)
    scatter_index=uspan_uv.unsqueeze(-1)+scatter_index

    N_all=torch.zeros((Nu_uv.size(0),Nu_uv.size(1),30),device=device)
    # print(143,scatter_index.dtype,Nu_uv.dtype,N_all.dtype)
    N_all=N_all.scatter_(2,scatter_index,Nu_uv)
    N_mask=points_mask.unsqueeze(-1).expand(-1,-1,30)

    N_all=torch.masked_fill(N_all,~N_mask.bool(),0)
        # N_all[s,uspan_uv[s,:]]
    ################################################################################
    ################################################################################

    ##### lstsq method #####
    # batch_size=points.shape[0]
    # all_loss=[]
    # for bat in range(batch_size):
    #     N_=N_all[bat,:,:knots_len[bat]-p-1]
    #     points_=points[bat]
    #     # solution_,loss_part,__,___=torch.linalg.lstsq(N_.T@N_,N_.T@points_)
    #     solution_=torch.linalg.pinv(N_)@points_
    #     residuals=points_-N_@solution_

    #     residuals=torch.masked_fill(residuals,~(points_mask.unsqueeze(-1).expand(-1,-1,points.shape[-1])).bool(),0)
    #     loss=torch.linalg.norm(residuals)
    #     all_loss.append(loss)
    # loss=torch.stack(all_loss)
    # loss=torch.sum(loss)
    # solution=None
    ##### lstsq method end #####

    # solution,loss,_,__=torch.linalg.lstsq(N_all,points)

    # assert(not points.isnan().any())
    # assert(not N_all.isnan().any())

    #### pinv method #####
    solution=torch.linalg.pinv(N_all)@points
    residuals=points-N_all@solution

    residuals=torch.masked_fill(residuals,~(points_mask.unsqueeze(-1).expand(-1,-1,points.shape[-1])).bool(),0)

    # loss=torch.sum(torch.square(residuals))
    # loss=torch.linalg.norm(residuals)
    # loss=torch.abs(residuals).sum()
    # loss=torch.pow(residuals,2).max(dim=-1)[0].sum()
    loss=torch.linalg.norm(residuals,dim=-1).max(dim=-1)[0].sum()
    # loss=torch.linalg.norm(residuals,dim=-1).max(dim=-1)[0].sum()

    #### pinv method end #####

    return loss,solution
    # return torch.functional.F.mse_loss((N_all@solution),points)

def getCtrlPts2(p,input,reduce=True):
    # input will be of dimension (batch_size, m+1, n+1, dimension)
    # 1 batch
    params,points,points_mask,knot_u,knots_len= input
    device=params.device

    params=torch.masked_fill(params,~points_mask.bool(),0)
    points=torch.masked_fill(points,~(points_mask.unsqueeze(-1).expand(-1,-1,points.shape[-1])).bool(),0)

    U=knot_u
    # U_c = torch.cumsum(torch.where(knot_u<0.0, knot_u*0+1e-4, knot_u), dim=1)
    # U = (U_c - U_c[:,0].unsqueeze(-1)) / (U_c[:,-1].unsqueeze(-1) - U_c[:,0].unsqueeze(-1))

    if torch.isnan(U).any():
        # print(U_c)
        print(knot_u)

    u = params.unsqueeze(1)
    uspan_uv = torch.stack([torch.min(torch.where((u[s] - U[s,p:].unsqueeze(1))>1e-8, u[s] - U[s,p:].unsqueeze(1), (u[s] - U[s,p:].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+p for s in range(U.size(0))])

    # torch._assert(torch.all(uspan_uv<25),"uspan out of bound")
    # print('uspan_uv:',uspan_uv)
    # print((u[0] - U[0,p:].unsqueeze(1))[...,35])

    u = u.squeeze(1)
    Ni = [u*0 for i in range(p+1)]
    Ni[0] = u*0 + 1
    for k in range(1,p+1):
        saved = (u)*0.0
        for r in range(k):
            UList1 = torch.stack([U[s,uspan_uv[s,:] + r + 1] for s in range(U.size(0))])
            UList2 = torch.stack([U[s,uspan_uv[s,:] + 1 - k + r] for s in range(U.size(0))])

            dU=((UList1 - u) + (u - UList2))
            dU_=torch.where(dU==0.0, u*0+1e-4, dU)

            temp = Ni[r]/dU_
            temp = torch.where(dU==0.0, u*0+1e-4, temp)
            Ni[r] = saved + (UList1 - u)*temp
            saved = (u - UList2)*temp
        Ni[k] = saved

    Nu_uv = torch.stack(Ni).permute(1,2,0)
    # torch.linalg.lstsq()
    scatter_index=torch.arange(-p,1,device=device).unsqueeze(0).unsqueeze(0).expand(uspan_uv.shape[0],uspan_uv.shape[1],-1)
    scatter_index=uspan_uv.unsqueeze(-1)+scatter_index

    N_all=torch.zeros((Nu_uv.size(0),Nu_uv.size(1),30),device=device)
    # print(143,scatter_index.dtype,Nu_uv.dtype,N_all.dtype)
    N_all=N_all.scatter_(2,scatter_index,Nu_uv)
    N_mask=points_mask.unsqueeze(-1).expand(-1,-1,30)

    N_all=torch.masked_fill(N_all,~N_mask.bool(),0)
        # N_all[s,uspan_uv[s,:]]
    ################################################################################
    ################################################################################

    ##### lstsq method #####
    # batch_size=points.shape[0]
    # all_loss=[]
    # for bat in range(batch_size):
    #     N_=N_all[bat,:,:knots_len[bat]-p-1]
    #     points_=points[bat]
    #     # solution_,loss_part,__,___=torch.linalg.lstsq(N_.T@N_,N_.T@points_)
    #     solution_=torch.linalg.pinv(N_)@points_
    #     residuals=points_-N_@solution_

    #     residuals=torch.masked_fill(residuals,~(points_mask.unsqueeze(-1).expand(-1,-1,points.shape[-1])).bool(),0)
    #     loss=torch.linalg.norm(residuals)
    #     all_loss.append(loss)
    # loss=torch.stack(all_loss)
    # loss=torch.sum(loss)
    # solution=None
    ##### lstsq method end #####

    # solution,loss,_,__=torch.linalg.lstsq(N_all,points)

    # assert(not points.isnan().any())
    # assert(not N_all.isnan().any())

    #### pinv method #####
    solution=torch.linalg.pinv(N_all)@points
    pred_points=N_all@solution
    residuals=points-pred_points

    residuals=torch.masked_fill(residuals,~(points_mask.unsqueeze(-1).expand(-1,-1,points.shape[-1])).bool(),0)

    # loss=torch.sum(torch.square(residuals))
    # loss=torch.linalg.norm(residuals)
    # loss=torch.abs(residuals).sum()
    # loss=torch.pow(residuals,2).max(dim=-1)[0].sum()
    loss=torch.linalg.norm(residuals,dim=-1)

    h_loss=Hausdorff_distance_batch_mask(pred_points,points,points_mask,points_mask)

    if reduce:
        loss1=loss.max(dim=-1)[0].sum()
        loss2=(loss.sum(dim=-1)/points_mask.sum(dim=-1)).sum()
        # loss2=loss.mean(dim=-1).sum()
        # loss=torch.linalg.norm(residuals,dim=-1).max(dim=-1)[0].sum()
        h_loss=h_loss.sum()
    else:
        loss1=loss.max(dim=-1)[0]
        # loss2=loss.mean(dim=-1)
        loss2=(loss.sum(dim=-1)/points_mask.sum(dim=-1))


    #### pinv method end #####

    return loss1,loss2,h_loss,solution
    # return torch.functional.F.mse_loss((N_all@solution),points)

def Hausdorff_distance_batch(a : torch.Tensor, b : torch.Tensor):
    """
    Params:
        a: (bs, sz1, 2), only bs = 1 used ? 
        b: (bs, sz2, 2)
    Returns:
        hdis: a scale, as bs = 1 is set 
    """
    # print("\nstart")
    # print(a[..., :10])
    # print(b[..., :10])
    # print("end\n")
    # assert a.shape == b.shape
    bs = a.size(0)
    sz1 = a.size(-2)
    sz2 = b.size(-2)

    expand_a = a.unsqueeze(dim=-2).expand(-1, -1, sz2,-1)
    expand_b = b.unsqueeze(dim=-3).expand(-1, sz1, -1, -1)
    delta = torch.linalg.norm(expand_a - expand_b, dim=-1)
    ret1 = delta.min(dim=-2)[0].max(dim=-1)[0]
    ret2 = delta.min(dim=-1)[0].max(dim=-1)[0]
    hdis = torch.where(ret1 > ret2, ret1, ret2)
    return hdis

def Hausdorff_distance_batch_mask(a : torch.Tensor, b : torch.Tensor,a_mask:torch.Tensor,b_mask:torch.Tensor):
    """
    Params:
        a: (bs, sz1, 2), only bs = 1 used ? 
        b: (bs, sz2, 2)
    Returns:
        hdis: a scale, as bs = 1 is set 
    """
    # print("\nstart")
    # print(a[..., :10])
    # print(b[..., :10])
    # print("end\n")
    # assert a.shape == b.shape
    bs = a.size(0)
    sz1 = a.size(-2)
    sz2 = b.size(-2)

    expand_a = a.unsqueeze(dim=-2).expand(-1, -1, sz2,-1)
    expand_b = b.unsqueeze(dim=-3).expand(-1, sz1, -1, -1)

    expand_a_mask=a_mask.unsqueeze(dim=-1).expand(-1,-1,sz2)
    expand_b_mask=b_mask.unsqueeze(dim=-2).expand(-1,sz1,-1)

    delta = torch.linalg.norm(expand_a - expand_b, dim=-1)
    delta_mask=torch.logical_and(expand_a_mask,expand_b_mask)
    delta=torch.masked_fill(delta,torch.logical_not(delta_mask),torch.finfo(torch.float32).max)
    ret1 = torch.masked_fill(delta.min(dim=-2)[0],torch.logical_not(b_mask),0).max(dim=-1)[0]
    ret2 = torch.masked_fill(delta.min(dim=-1)[0],torch.logical_not(a_mask),0).max(dim=-1)[0]
    hdis = torch.where(ret1 > ret2, ret1, ret2)
    return hdis
