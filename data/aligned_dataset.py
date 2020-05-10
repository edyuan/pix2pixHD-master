import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import torch

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))
        
        if self.A_paths[0].split('/')[-1][0] == '0': # new dataset format
             opt.two_and_half_D = True
             opt.input_nc = max([int(a.split('_')[-1][0]) for a in self.A_paths])
             os.path.join(opt.dataroot, opt.phase + dir_A + '')
             unique_paths = [p for p in self.A_paths if '_1.' in p]

             self.A_paths = []
             for p in unique_paths:
                 temp = []
                 for i in range(opt.input_nc):
                     newpath = '_'.join(p.split('_')[:-1]) + '_' + str(i+1) + '.jpg'
                     temp.append(newpath)                 
    
                 self.A_paths.append(temp)
             #for p in self.A_paths:
             #    print(p)

        ### input B (real images)
        dir_B = '_B' if self.opt.label_nc == 0 else '_img'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
        self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.B_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]
        if not self.opt.two_and_half_D: # place in list if not 2.5D to match format
            A_path = [A_path]

        A_tensor_list = []
        for A_slice_path in A_path:
            A = Image.open(A_slice_path)        
            params = get_params(self.opt, A.size)
            if self.opt.label_nc == 0:
                transform_A = get_transform(self.opt, params)
                A_tensor_list.append(transform_A(A.convert('RGB'))) #A_tensor shape is 3, 672, 1024
            else:
                transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
                A_tensor_list.append(transform_A(A) * 255.0)

        # create tensor
        A_tensor = torch.zeros([self.opt.input_nc, A_tensor_list[0].shape[1], A_tensor_list[0].shape[2]])        

        if self.opt.input_nc != len(A_tensor_list):
            assert len(A_tensor_list) == 1, "# of images not 1 and mismatch with opt.input_nc"
            for i in range(self.opt.input_nc):
                A_tensor[i,:,:] = A_tensor_list[0][0,:,:]
        else:
            for i, a_tensor in enumerate(A_tensor_list):
                A_tensor[i,:,:] = a_tensor[0,:,:]

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        B_path = self.B_paths[index]   
        B = Image.open(B_path).convert('RGB')
        transform_B = get_transform(self.opt, params)      
        B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
