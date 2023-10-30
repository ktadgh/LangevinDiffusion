#include <iostream>
#include <vector>
#include <string>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <chrono>
#include <functional>
#include <cmath>
#include <iterator> // Add this line
#include <torch/extension.h>
#include <pybind11/pybind11.h>

using namespace std;
using namespace std::placeholders;  // for _1, _2, _3...
using namespace torch::indexing;
using Function = function<torch::Tensor(torch::Tensor)>;

torch::Tensor jacobian(std::vector<Function>& funcs, torch::Tensor& x)
{ 
    torch::Tensor j = torch::ones({funcs.size(), x.sizes()[0]},{torch::kFloat64});
    int i {0};
    torch::Tensor ones = torch::ones_like(x);
    for (Function func : funcs)
    {
        auto b = func(x);
        torch::Tensor deriv = torch::autograd::grad({b}, {x}, /*grad_outputs=*/{ones}, /*create_graph=*/true, /*retain_graph=*/true,/*allow_unused=*/false)[0];
        j[i] = deriv;
        i++;
    }
    //TODO: Need to triple check that outputs and gradients are correct
    return j;
    
}

torch::Tensor fast_jacobian(std::vector<Function>& funcs, torch::Tensor& x)
{
    std::vector<torch::Tensor> vec;
    auto start = std::chrono::high_resolution_clock::now();
    //torch::Tensor j = torch::ones({funcs.size(), x.sizes()[0]},{torch::kFloat64});
    int i {0};
    torch::Tensor ones = torch::ones_like(x);
    for (Function func : funcs)
    {
        x = x.detach();
        x.set_requires_grad(true);
        auto b = func(x);
        b.backward({},false);
        vec.push_back(x.grad());
        i++;
    }

    //TODO: Need to triple check that outputs and gradients are correct
    torch::Tensor j = torch::stack(vec);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    //std::cout << "FAST JACOBIAN DURATION = " << duration.count() << "Microseconds" << endl;
    return j;
    
}

torch::Tensor evG(std::vector<Function>& funcs, torch::Tensor& x)
{
    torch::Tensor G = torch::ones({funcs.size()},{torch::kFloat64});
    int i {0};
    for (Function func : funcs)
    {
        auto b = func(x);
        G[i] = b.squeeze();
        i++;
    }
    return G;
}

// Next steps: Include the "simulation" function
void rattle_step(torch::Tensor& x_col, torch::Tensor& v1_col,float h,torch::Tensor& M,std::vector<Function>& gs, float e)
{
    torch::Tensor M1 = torch::inverse(M);
    torch::Tensor DV_col = torch::zeros_like(x_col,{torch::kFloat64});

    // doing newton-raphson iterations
    torch::Tensor x2 = (x_col + h * v1_col - 0.5*(pow(h,2))* torch::matmul(M1 , DV_col));
    torch::Tensor Q_col = x2;
    torch::Tensor Q = Q_col.squeeze();
    torch::Tensor x_col_sq = x_col.squeeze();
    ////std::cout << "x_col dtype " << x_col.dtype() << endl;
    x_col_sq.set_requires_grad(true);
    //x_col_sq.retain_grad();
    torch::Tensor J1 = fast_jacobian(gs, x_col_sq);
    torch::Tensor Q_sq = Q.squeeze();
    torch::Tensor dL;
    #pragma GCC ivdep
    for (int its = 0; its < 3; its++)
    {
        Q_sq = Q.squeeze();
        ////std::cout << "Q_sq dtype " << Q_sq.dtype() << endl;
        Q_sq.set_requires_grad(true);
        //Q_sq.retain_grad();
        // torch::Tensor J2 = fast_jacobian(gs, Q_sq);
        // torch::Tensor R = torch::matmul(J2 ,torch::matmul(M1, J1.t()));
        // dL = torch::matmul(torch::inverse(R), evG(gs,Q));
        dL = torch::matmul(torch::inverse(torch::matmul(fast_jacobian(gs, Q_sq) ,torch::matmul(M1, J1.t()))), evG(gs,Q));
        Q = Q - (M1 , torch::matmul(J1.t() , dL));
    }
    ////std::cout << "CHECK 3 " << endl;

    Q = Q.t();
    
    torch::Tensor v1_half = (Q.unsqueeze(1) - x_col)/h;
    x_col = Q;
    x_col_sq = x_col.squeeze();
    x_col_sq.retain_grad();

    J1 = fast_jacobian(gs, x_col_sq);
    // getting the level
    Q_sq = Q_col.squeeze();
    ////std::cout << "Q_sq dtype " << Q_sq.dtype() << endl;
    Q_sq.set_requires_grad(true);
    Q_sq.retain_grad();
    torch::Tensor J2 = fast_jacobian(gs, Q_sq);
    torch::Tensor P = torch::matmul(J1, torch::matmul(M1 , J1.t()));
    torch::Tensor part = (2/h * v1_half - torch::matmul(M1 ,DV_col));
    torch::Tensor T = torch::matmul(J1 , part);

    // solving the linear system - we don't have this in fcking c++
    
    torch::Tensor L = torch::linalg::solve(P,T, true);
    // should defintely check that this is correct

    v1_col = v1_half - h/2 * DV_col - h/2 * torch::matmul(J2.t(),L);
    x_col = x_col.unsqueeze(1);
}


void gBAOAB_step_exact(torch::Tensor& q_init, torch::Tensor& p_init, Function& F, std::vector<Function>& gs, float h, torch::Tensor& M, torch::Tensor& gamma, float k, float kr, float e)
{
    //std::cout << "TEST EXACT 1" << endl;
    torch::Tensor M1 = M;
    torch::Tensor R = torch::randn(q_init.numel(),{torch::kFloat64});
    torch::Tensor p = p_init;
    torch::Tensor q = q_init;
    torch::Tensor a2 = torch::exp(-gamma * h);
    torch::Tensor b2 = torch::sqrt(k*(1-a2.pow(2)));
    //std::cout << "TEST EXACT 2" << endl;


    // doing the initial p-update
    // note to self - should really just change the definition of jacobian to squeeze everything
    torch::Tensor q_sq = q.squeeze();
    torch::Tensor G = fast_jacobian(gs, q_sq);
    torch::Tensor GT = torch::transpose(G,0,1);
    torch::Tensor inter1 = torch::matmul(G,torch::matmul(M1, GT));
    torch::Tensor inter2 = torch::matmul(GT, torch::inverse(inter1));
    torch::Tensor inter3 = torch::matmul(G,M1);
    torch::Tensor L1 = torch::eye(q_init.numel(),{torch::kFloat64}) - torch::matmul(inter2, inter3);
    p = p - h/2 * torch::matmul(L1, F(q));
    //std::cout << "TEST EXACT 3" << endl;

    torch::Tensor pun = p.unsqueeze(1);
    torch::Tensor qun = q.unsqueeze(1);
    
    // doing the first RATTLE step
    #pragma GCC ivdep
    for(int i = 0; i < kr; i++)
    {
        // rattle_step updates p and q in place
        rattle_step(qun,pun,h/2 * kr, M, gs, e);
    }
    q = qun.squeeze();
    p = pun.squeeze();
    // the second p-update 
    q_sq = q.squeeze();
    G = fast_jacobian(gs, q_sq);
    GT = torch::transpose(G,0,1);
    inter1 = torch::matmul(G,torch::matmul(M1, GT));
    inter2 = torch::matmul(GT, torch::inverse(inter1));
    inter3 = torch::matmul(G,M1);
    torch::Tensor L2 = torch::eye(q_init.numel(),{torch::kFloat64}) - torch::matmul(inter2, inter3);
    p =  p - h/2 * torch::matmul(L2 ,F(q));


    // doing the rattle step
    pun = p.unsqueeze(1);
    qun = q.unsqueeze(1);

    #pragma GCC ivdep
    for(int i =0; i < kr; i++)
    {
        rattle_step(qun,pun,h/2 * kr, M, gs, e);
    }
    
    q = qun.squeeze();
    p = pun.squeeze();
    // //std::cout <<"check 5"<<endl;
    // the final p update
    q_sq = q.squeeze();
    G = fast_jacobian(gs, q_sq);
    GT = torch::transpose(G,0,1);
    inter1 = torch::matmul(G,torch::matmul(M1, GT));
    inter2 = torch::matmul(GT, torch::inverse(inter1));
    inter3 = torch::matmul(G,M1);
    torch::Tensor L3 = torch::eye(q_init.numel(),{torch::kFloat64}) - torch::matmul(inter2, inter3);
    p = p - h/2 * torch::matmul(L3 ,F(q));
    q_init = q;
    p_init = p;
}


void gBAOAB_integrator(torch::Tensor& q_init, torch::Tensor& p_init, Function F, std::vector<Function>& gs, float h, torch::Tensor M, torch::Tensor gamma, float k, float steps, float kr, float e)
{
    //std::cout << "TEST gb1" << endl;
    std::vector<int64_t> vec = {steps, q_init.numel()};
    torch::Tensor positions = torch::zeros(vec,{torch::kFloat64});
    torch::Tensor velocities = torch::zeros(vec,{torch::kFloat64});
    torch::Tensor q = q_init;
    torch::Tensor p = p_init;
    #pragma GCC ivdep
    for(int i = 0; i < steps; i++)
    {
        //std::cout << "gbaoab integrator step"<< i << endl;
        gBAOAB_step_exact(q,p,F,gs,h,M,gamma,k,kr,e);
    }
    // should really change this to be MORE EFFICIENT!!
    q_init = q;
    p_init= p;
}




// Make a vectorized gBAOAB multi-integrator           
// torch::Tensor multi_gBAOAB_integrator(torch::Tensor q_init, torch::Tensor p_init, Function F, std::vector<Function>& gs, float h, torch::Tensor M, torch::Tensor gamma, float k, float steps, float kr, float e)
// {
//     std::vector<int64_t> vec = {10,steps, q_init.numel()};
//     torch::Tensor output = torch::zeros(vec,{torch::kFloat64});
//     #pragma GCC ivdep
//     for(int j = 0; j < 10; j ++)
//     {;
//         std::vector<int64_t> vec1 = {steps, q_init.numel()};
//         torch::Tensor positions = torch::zeros(vec1,{torch::kFloat64});
//         torch::Tensor q = q_init;
//         torch::Tensor p = p_init;
//         for(int i = 0; i < steps; i++)
//         {
//             gBAOAB_step_exact(q,p,F,gs,h,M,gamma,k,kr,e);
//         }
//         output[j] = positions;
//     }
//     return output;
//     // should really change this to return both
// }



torch::Tensor F(torch::Tensor x)
{
    return 2*x*2*(x.pow(2) -1);
}

torch::Tensor g1(torch::Tensor x)
{
    return x.pow(2).sum() -1;
}


// a function that takes a tensor, an initial tenor, and two indexes and returns the distance between them
torch::Tensor my_length_constraint(torch::Tensor x, torch::Tensor x_init, int i, int j)
{
    return pow((x[3*i]- x[3*j]),2) +pow((x[3*i+1]- x[3*j+1]),2) + pow((x[3*i+2]- x[3*j+2]),2) - pow((x_init[3*i]- x_init[3*j]),2) - pow((x_init[3*i+1]- x_init[3*j+1]),2) - pow((x_init[3*i+2]- x_init[3*j+2]),2);
}


Function length_constraint(torch::Tensor x_init, int i, int j)
{
    auto f = std::bind(my_length_constraint,std::placeholders::_1, x_init, i,j);
    return f;
}


// cotangent projection function..... projecting the score function onto the tangent space of the manifold.
torch::Tensor my_cotangent_projection(std::vector<Function> gs, torch::Tensor x)
{
    x.set_requires_grad(true);
    x.retain_grad();
    torch::Tensor G = jacobian(gs,x);
    torch::Tensor M = torch::eye(G.sizes()[1],{torch::kFloat64});
    torch::Tensor L0 = torch::matmul(G,torch::matmul(M,G.t()));
    torch::Tensor L1 = torch::matmul(G.t(),torch::inverse(L0));
    torch::Tensor L2 = torch::matmul(torch::matmul(L1, G), torch::inverse(M));
    torch::Tensor L = torch::eye(G.sizes()[1],{torch::kFloat64}) -L2;
    return L;
}



Function cotangent_projection(std::vector<Function> gs)
{
    auto f = std::bind(my_cotangent_projection,gs,std::placeholders::_1);
    return f;
}


std::vector<torch::Tensor> lengths(std::vector<std::tuple<int,int>> bones,torch::Tensor x_init)
{
    torch::Tensor pose = torch::reshape(x_init, {19,3});
    std::vector<torch::Tensor> lengths;
    for (std::tuple<int,int> bone : bones)
    {
        int first = std::get<0>(bone); 
        int second = std::get<1>(bone);
        torch::Tensor l = pose[first] - pose[second];
        lengths.push_back(l);
    }
    return lengths;

}

torch::Tensor angles_to_joints(torch::Tensor angles, std::vector<torch::Tensor> lengths, std::vector<std::tuple<int,int>> bones)
{
    torch::Tensor pose = torch::zeros({19,3}, torch::kFloat64);
    pose[0] = torch::tensor({0,0,2});
    for (int i = 0; i < bones.size(); i ++)
    {
        std::tuple<int,int> bone = bones[i];
        int first = std::get<0>(bone); 
        int second = std::get<1>(bone);
        torch::Tensor r = lengths[i];
        torch::Tensor phi = angles[i][0];
        torch::Tensor theta = angles[i][1];
        float x = (r * torch::sin(theta) * torch::cos(phi)).item<float>();
        float y = (r * torch::sin(theta) * torch::sin(phi)).item<float>();
        float z = (r * torch::cos(theta)).item<float>();

        pose[second] = pose[first] + torch::tensor({x, y, z});

    }
    return pose;
}

torch::Tensor uniform_generator(std::vector<std::tuple<int,int>> bones,torch::Tensor x_init)
{
    torch::Tensor random_angles = torch::vstack({torch::rand(18,{torch::kFloat64})*M_PI*2, torch::acos(2*torch::rand(18,{torch::kFloat64})-1)}).t();
    std::vector<torch::Tensor> lengths1 = lengths(bones, x_init);
    return angles_to_joints(random_angles, lengths1, bones);
}


// creating the neural network
struct GFP : torch::nn::Module
{
    GFP(int embed_dim, int scale )
    {
        W = register_parameter("W", torch::randn({embed_dim/2},{torch::kFloat64}));
        ////std::cout << "W dtype " << W.dtype() << endl;
        W.set_requires_grad(false);
    }

    GFP() = default;
    torch::Tensor forward(torch::Tensor x){
        torch::Tensor x_proj = x.unsqueeze(1) * W.unsqueeze(0) * (2 * M_PI);
        return torch::cat({torch::sin(x_proj).to(torch::kFloat64), torch::cos(x_proj).to(torch::kFloat64)}, -1);
    }
    torch::Tensor W; // allows outside access, not even sure if I need this line

};


struct Net : torch::nn::Module
{
    Net(int embed_dim)
    {
        torch::manual_seed(0);
        embed = register_module("embed",std::make_shared<GFP>(embed_dim, 1.0/3.0));
        embed2 = register_module("embed2",torch::nn::Linear(embed_dim, embed_dim));
        lin_embed = register_module("lin_embed", torch::nn::Linear(embed_dim, 150));
        lin_embed2 = register_module("lin_embed2", torch::nn::Linear(embed_dim, 150));     
        lin1 = register_module("lin1",torch::nn::Linear(57,150));
        lin2 = register_module("lin2",torch::nn::Linear(150, 57));
        lin3 = register_module("lin3",torch::nn::Linear(150,150));
        lin4 = register_module("lin4",torch::nn::Linear(150, 57));
        act = register_module("act",torch::nn::SiLU());
    }

    torch::nn::ModuleHolder<GFP> embed;
    torch::nn::Linear embed2 = nullptr;
    torch::nn::Linear lin_embed = nullptr;
    torch::nn::Linear lin_embed2 = nullptr;
    torch::nn::Linear lin1 = nullptr;
    torch::nn::Linear lin2 = nullptr;
    torch::nn::Linear lin3 = nullptr;
    torch::nn::Linear lin4 = nullptr;
    torch::nn::SiLU act;


    torch::Tensor forward(torch::Tensor x, torch::Tensor t, torch::Tensor L)
    {
        embed2->to(torch::kDouble);
        lin_embed->to(torch::kDouble);
        lin_embed2->to(torch::kDouble);
        lin1->to(torch::kDouble);
        lin2->to(torch::kDouble);
        lin3->to(torch::kDouble);
        lin4->to(torch::kDouble);
        act->to(torch::kDouble);

        //TODO - can I make this batchable?
        ////std::cout << "Shape of x" << x.sizes() << endl;
        x = torch::unsqueeze(x, 0);
        torch::Tensor l = torch::zeros_like(x,torch::kFloat64);
        l.index({Slice(), 0}).copy_(x.index({Slice(), 0}));
        l.index({Slice(), 1}).copy_(x.index({Slice(), 1}));
        l.index({Slice(), 0}).copy_(-torch::ones_like(x.index({Slice(), 2}),torch::kFloat64) * 2 + x.index({Slice(), 2}));


        x = x-l;
        ////std::cout << "Test 1" <<  endl;
        ////std::cout << "Test 1.1" << t.dtype()<<  endl;
        torch::Tensor t1 = t.to(torch::kFloat64);
        torch::Tensor emb = embed(t1);
        ////std::cout << "Test 1.5" << emb.dtype()<<  endl;
        torch::Tensor emb2 = embed2(embed(t1));
        ////std::cout << "test 1.6" << emb2.dtype() << endl;
        torch::Tensor embedded = act(embed2(embed(t1)));
        ////std::cout << "Test 2" <<  endl;
        torch::Tensor h = lin1(x);
        h = h + lin_embed(embedded);
        ////std::cout << "Shape of h" << h.sizes() << endl;
        h = act(lin3(h) + lin_embed2(embedded));
        h = lin4(h);
        h = torch::unsqueeze(torch::matmul(L,torch::squeeze(h)),0);

        torch::Tensor l2 = torch::zeros_like(h,torch::kFloat64);
        l2.index({Slice(), 0}).copy_(h.index({Slice(), 0}));
        l2.index({Slice(), 1}).copy_(h.index({Slice(), 1}));
        l2.index({Slice(), 2}).copy_(h.index({Slice(), 2}));
        h = h-l2;

        return torch::squeeze(h);
    }
};

torch::Tensor net_jacobian(Net& net, torch::Tensor x,torch::Tensor& random_t,torch::Tensor& L)
{
    torch::Tensor y = net.forward(x, random_t, L); // TODO BUT L DEPENDS ON X?
    // TODO: have an assert here, this should only really be needed for 1d functions
    torch::Tensor j = torch::ones({y.numel(), x.sizes()[0]},{torch::kFloat64});

    int i {0};
    torch::Tensor ones = torch::ones_like(x);
    for (int z =0; z < y.numel(); z++)
    {
        y = net.forward(x, random_t, L);
        auto b = y[z];
        torch::Tensor deriv = torch::autograd::grad({b}, {x}, /*grad_outputs=*/{ones}, /*retain_graph=*/true,/*create_graph=*/true, /*allow_unused=*/false)[0];
        j[z] = deriv;
        i++;
    }
    //TODO: Need to triple check that outputs and gradients are correct
    ////std::cout << "net jacobian is fine" << endl; 
    return j;
}

torch::Tensor F_zero(torch::Tensor x)
{
    return torch::zeros_like(x,torch::kFloat64);
}


torch::Tensor loss_fn(Net net, torch::Tensor xs, torch::Tensor random_t, std::vector<std::tuple<int, int>> bones)
{
    torch::Tensor l = torch::zeros(1,torch::kFloat64);
    l.set_requires_grad(true);
    torch::Tensor tr = torch::zeros(0,torch::kFloat64);
    torch::Tensor s_tr = torch::zeros(0,torch::kFloat64);

    // iterating through the batches
    for (int z =0; z< xs.sizes()[0]; ++z)
    {
        torch::Tensor x = xs[z];
        torch::Tensor q = torch::squeeze(x);
        q.set_requires_grad(true);

        std::vector<Function> constraints;

        // setting the constraints according to the inital pose
        for (const auto& bone : bones)
        {
            int joint1 = std::get<0>(bone);
            int joint2 = std::get<1>(bone);
            Function g = length_constraint(x, joint1, joint2);
            constraints.push_back(g);
        }
        
        // now I need to noise my thing according to the constraints.
        // getting the projection matrix?

        //TODO the high should be programmable, as should the stepsize
        Function L_fn = cotangent_projection(constraints);
        torch::Tensor positions = xs;


        // set what q is equal to here
        torch::Tensor L = L_fn(q); // defining the projection matrix
        torch::Tensor score = net.forward(q,random_t, L);
        ////std::cout << " SCOREscore" << score << endl;


        // maybe I need to define the function similarly, so that ...
        tr = s_tr +torch::trace(torch::squeeze(net_jacobian(net, q, random_t, L)));
        l = l + torch::tensor({0.5},torch::kFloat64) * torch::norm(score).pow(2) + torch::trace(torch::squeeze(net_jacobian(net, q,random_t,L)));
        // I need a new jacobian function which takes one function instead of a list! Overload it

        ////std::cout << "norm score" <<torch::tensor({0.5},torch::kFloat64) * torch::sqrt(score.pow(2).sum()).pow(2) << endl;
    }

    //std::cout << "loss " << l << endl;
    return l;
}


// in main I will have a list of tuples of bones, defined.


torch::Tensor train_net(torch::Tensor data_tensor, torch::Tensor random_t, torch::Tensor tbones)
{
    pybind11::gil_scoped_release no_gil;
    int embed_dim = 32;
    int n_epochs = 3;
    Net model(embed_dim);

    torch::data::datasets::TensorDataset dataset(data_tensor);
    auto data_loader = torch::data::make_data_loader(std::move(dataset));

    // taking bones and changing to an std vec
    std::vector<std::tuple<int, int>> bones;
    for(int i = 0; i < tbones.sizes()[0]; i++)
    {
        int a = tbones[i][0].item<int>();
        int b = tbones[i][1].item<int>();
        bones.push_back(std::make_tuple(a, b));
    }
    torch::optim::SGD optimizer(model.parameters(),0.00001);
    torch::Tensor losses = torch::zeros(n_epochs);
    for(int epoch = 0; epoch < n_epochs; ++epoch)
    {
        model.train();
        for(auto& batch: *data_loader)
        {
            std::vector<torch::Tensor> inputs;
            for(auto& example: batch)
            {
                inputs.push_back(example.data);
            }
            torch::Tensor input_tensor = torch::stack(inputs);
            ////std::cout << "Input Tensor = " << input_tensor.sizes();
            //TODO: print model parameters
            //std::cout << "Model Parameters" << model.parameters() << endl;
            torch::Tensor loss = loss_fn(model, input_tensor,random_t, bones);


            //backpropogation
            losses[epoch] = loss.item();
            loss.backward();

            //update parameters
            optimizer.step();
        }
    }
    return losses;
}

torch::Tensor noise(torch::Tensor xs,torch::Tensor random_ts, torch::Tensor tbones)
{
    pybind11::gil_scoped_release no_gil;
    std::vector<std::tuple<int, int>> bones;
    for(int i = 0; i < tbones.sizes()[0]; i++)
    {
        int a = tbones[i][0].item<int>();
        int b = tbones[i][1].item<int>();
        bones.push_back(std::make_tuple(a, b));
    }

    // creating the output vector
    torch::Tensor pos = torch::zeros_like(xs,{torch::kFloat64});

    // noising
    for (int z =0; z< xs.sizes()[0]; ++z)
    {
        //std::cout << "TEST 1" << endl;
        torch::Tensor x = xs[z];
        torch::Tensor q = torch::squeeze(x);
        q.set_requires_grad(true); // I think I need this so that the jacobian function works, should remove it if not though TODO

        //std::cout << "TEST 2" << endl;
        std::vector<Function> constraints;
        // setting the constraints according to the inital pose
        for (const auto& bone : bones)
        {
            int joint1 = std::get<0>(bone);
            int joint2 = std::get<1>(bone);
            Function g = length_constraint(x, joint1, joint2);
            constraints.push_back(g);
        }
        
        // now I need to noise my thing according to the constraints.
        // getting the projection matrix?

        //TODO the high should be programmable, as should the stepsize
        //std::cout << "TEST 3" << endl;
        torch::Tensor random_t = random_ts[z]; // uniform between 0 and 1
        int steps =1+ (random_t / 0.01).to(torch::kInt).item<int>();
        std::cout << "STEPS = " << steps <<  endl;
        torch::Tensor M = torch::eye(x.sizes()[0],{torch::kFloat64});

        //TODO make stepsize programmable


        // something like this
        //std::cout << "TEST 4" << endl;
        torch::Tensor p_init = torch::zeros_like(q,torch::kFloat64);
        torch::manual_seed(0);
        gBAOAB_integrator(q,p_init, F_zero, constraints, 0.01, M, torch::tensor({1}), 1, steps, 3, 1);
        pos[z] = q;
    }
    return pos;
}

torch::Tensor loss_test(torch::Tensor xs, torch::Tensor random_ts, torch::Tensor tbones)
{   
    torch::manual_seed(0);
    pybind11::gil_scoped_release no_gil;
    int embed_dim = 32;
    Net model(embed_dim);
    model.to(torch::kFloat64);


    std::vector<std::tuple<int, int>> bones;
    for(int i = 0; i < tbones.sizes()[0]; i++)
    {
        int a = tbones[i][0].item<int>();
        int b = tbones[i][1].item<int>();
        bones.push_back(std::make_tuple(a, b));
    }
    
    torch::Tensor l = torch::zeros(1);
    l.set_requires_grad(true);
    torch::Tensor tr = torch::zeros(1,torch::kFloat64);
    torch::Tensor s_tr = torch::zeros(1,torch::kFloat64);
    // iterating through the batches

    for (int z =0; z< xs.sizes()[0]; ++z)
    {
        torch::Tensor random_t = random_ts[z].unsqueeze(0);
        torch::Tensor x = xs[z];
        torch::Tensor q_fin = torch::squeeze(x);
        q_fin.set_requires_grad(true);

        std::vector<Function> constraints;

        // setting the constraints according to the pose
        for (const auto& bone : bones)
        {
            int joint1 = std::get<0>(bone);
            int joint2 = std::get<1>(bone);
            Function g = length_constraint(x, joint1, joint2);
            constraints.push_back(g);
        }
        
        // now I need to noise my thing according to the constraints.
        // getting the projection matrix?

        //TODO the high should be programmable, as should the stepsize
        Function L_fn = cotangent_projection(constraints);

        // set what q is equal to here
        torch::Tensor L = L_fn(q_fin); // defining the projection matrix
        L.set_requires_grad(true);
        //L = torch::eye(q_fin.sizes()[0],torch::kFloat64);

        torch::Tensor score = model.forward(q_fin,random_t, L);
        ////std::cout << " SCOREscore" << score << endl;


        // maybe I need to define the function similarly, so that ...
        tr = s_tr +torch::trace(torch::squeeze(net_jacobian(model, q_fin, random_t, L)));
        l = l + torch::tensor({0.5},torch::kFloat64) * torch::norm(score).pow(2) + torch::trace(torch::squeeze(net_jacobian(model, q_fin,random_t,L)));
        // I need a new jacobian function which takes one function instead of a list! Overload it

        ////std::cout << "norm score" <<torch::tensor({0.5},torch::kFloat64) * torch::sqrt(score.pow(2).sum()).pow(2) << endl;
    }
    return l;
}


// torch::Tensor model_test(torch::Tensor xs, torch::Tensor tbones)
// {
//     torch::manual_seed(0);
//     pybind11::gil_scoped_release no_gil;
//     int embed_dim = 32;
//     Net model(embed_dim);
//     model.to(torch::kFloat64);
//     torch::Tensor l = torch::zeros(1);
//     l.set_requires_grad(true);
//     for (int z =0; z< xs.sizes()[0]; ++z)
//     {
//         torch::Tensor x = xs[z];
//         torch::Tensor q = torch::squeeze(x);
//         q.set_requires_grad(true);

//     }


// }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("train_net", &train_net, "Description of your function");
    m.def("noise", &noise, "Description of your function");
    m.def("loss_test", &loss_test, "Description of your function");
    // Add more function bindings as needed
}


int main()
{
    //torch::set_default_dtype(caffe2::TypeMeta::Id<float>());
    //torch::set_assert_no_internal_overlap(true);

    // doing a less quick test of the gbaoab_integrator function


    ////std::cout << "Position shape = " << position.sizes() << endl;

    int batch_size = 8;
    int batches = 100;
    int n_epochs = 10;

    //scalarToTensor::vector<char> f = get_the_bytes("path\\to\\test.pt");
    //torch::IValue x = torch::pickle_load(f);
    torch::Tensor data_tensor = torch::rand({4,57}).to(torch::kFloat64);
    torch::data::datasets::TensorDataset dataset(data_tensor);
    auto data_loader = torch::data::make_data_loader(std::move(dataset));
    
    Net model(32);
    std::vector<std::tuple<int, int>> bones;
    bones.push_back(std::make_tuple(1, 2));
    bones.push_back(std::make_tuple(3, 4));
    bones.push_back(std::make_tuple(4, 5));
    torch::optim::SGD optimizer(model.parameters(),0.00001);
    //TODO change ... learning rate?

    auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor tbones = torch::tensor({{1,2},{1,3},{1,4},{4,5},{5,6},{7,8},{8,9}});
    torch::Tensor random_t = torch::rand(data_tensor.sizes()[0]);
    torch::Tensor noised_data = noise(data_tensor, random_t, tbones);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    //std::cout << "DURATION = " << duration.count()/1000000 << endl;
    }
