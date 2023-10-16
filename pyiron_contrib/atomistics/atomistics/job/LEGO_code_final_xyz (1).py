#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyiron_base import TemplateJob, DataContainer
from pyiron_atomistics import Project
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure
import os
import os.path
import warnings
import datetime as dt


# In[2]:


class Remote_Control:
    def input_func(self):
        
        ##path -> defines the path where all the input files and generated files are kept
        ##element -> which element is used in the generated XYZ file in its 1st column as needed
        ##parameter -> Defines the name of the parameter files (one or more than one) in a comma seperated manner. 
                        ##Paramter files must be present inside the defined base path.
        ##header_file -> Defines the name of the header file that will be appeneded with the lego output.
        ##IMD_file_name -> Defines the name of the final IMD file that will be stored. 
                          ##This is a user defined name and can be anything as per the user's wish
        
        parameter_list = []
        
        path = input('Enter the base path : ')    
        print('the base path is : {}'.format(path))
        element = input('Enter Element name :')
        print('the element name is : {}'.format(element))
        warnings.warn('If there are more than 1 paramter file then please put the names of all the files in the below field seperated by comma', UserWarning)
        parameter = input ('Enter the parameter file name : ')
        print('the parameter name is : {}'.format(parameter))
        header_file = input('Enter the header file name : ' )
        print('the header file name is : {}'.format(header_file))
        IMD_file_name = input('Enter the IMD file name : ')
        print('The IMD file name is : {}'.format(IMD_file_name))
        
        for i in range(0,len(parameter.split(','))):
            parameter_list.append(parameter.split(',')[i].split()[0])
        #print(b)
        
        #return path,element,parameter,IMD_file_name,parameter_list
        return path,element,parameter_list,header_file,IMD_file_name
        
#         self.parameter = parameter
        
#     def input_func(self):
        
#         self.path = input('Enter the base path : ')
#         print('the base path is : {}'.format(self.path))
        
        


# In[ ]:


#motat = Remote_Control()


# In[ ]:


#a,b,c,d,e = mota.input_func()


# In[ ]:


c


# In[3]:


##This class generates a shell script in a dynamic way. Different paramter files has different names of the generated out files
##and hence its not possible to hard code it as it will crash after its executed due to different file name mismatches.Hence 
##the shell script needs to be made in a dynamic way by reading the names of the output files from the parameter(.confparm) files
##and appending those into the script. 

class LEGO_shell_script:
    
    def __init__(self,base_path,parameter_list, header_file,IMD_file_name):
        self.base_path = base_path
        self.parameter_list = parameter_list
        self.header_file = header_file
        #self.no_parameter = no_parameter
        self.IMD_file_name = IMD_file_name
        self.shell_script_name = 'lego'+'_'+'{}'.format(self.IMD_file_name.split('.')[0])+'.'+'sh'
        self.my_list_para = []
        self.my_new_list = []
        #self.LEGO_output_file = []
        
    def shell_script(self):
        
        #para = 'para'
#         FileHandler = open(self.path,"r")
#         self.my_list.append(FileHandler.readlines())
#         self.my_list = self.my_list[0]
#         self.my_new_list.extend(['#!/bin/bash\n','\n','current_path="{}"\n'.format(self.base_path),'present_path=$(pwd)\n',
#                                  '\n','cd $current_path\n','\n','$current_path/legoV3.1 $current_path/{}\n'.format(self.input_param),'\n',
#                                  '#mv \n','\n','cp $current_path/header.txt $current_path/{}\n'.format(self.IMD_file_name),'\n',
#                                  'cat 111.fcc >> {}\n'.format(self.IMD_file_name),'\n','chmod +rwx {}\n'.format(self.IMD_file_name),'\n'])
       
        
        self.my_new_list.extend(['#!/bin/bash\n','\n','current_path="{}"\n'.format(self.base_path),'present_path=$(pwd)\n',
                               '\n','cd $current_path\n','\n'])
        
        for i in range (0,len(self.parameter_list)):
            self.my_new_list.append('$current_path/lego $current_path/{}\n'.format(self.parameter_list[i]))
            self.my_new_list.append('\n')
            
        
        self.my_new_list.append('cp $current_path/{} $current_path/{}\n'.format(self.header_file,self.IMD_file_name))
        self.my_new_list.append('\n')
        
#         for i in os.listdir(self.base_path):
        
#             LEGO_output_file_creation_time = dt.datetime.fromtimestamp(os.path.getctime(os.path.join(self.base_path,'{}'.format(i)))).timestamp()
#             #LEGO_output_file_creation_time = LEGO_output_file_creation_time.split(' ')[-1].split(':')
#             current_time = dt.datetime.now().timestamp()
#             #current_time = current_time.split(' ')[-1].split(':')
        for i in self.parameter_list:
            FileHandler = open(os.path.join(self.base_path,'lego_{}.input'.format(i)),"r")
            self.my_list_para.append(FileHandler.readlines())
            #self.my_list_para = self.my_list_para[0]
            for j in self.my_list_para[-1]:
                if j.split()[0] == 'outfile':
                    #para.append(i.split()[-1])
                    self.para = j.split()[-1]
        
    
            if self.para.split('.')[-1] == 'fcc':   #and int(current_time)-int(LEGO_output_file_creation_time) < 120:
                self.my_new_list.append('cat {} >> {}\n'.format(self.para, self.IMD_file_name))
                self.my_new_list.append('\n')
        
        with open(os.path.join(self.base_path, '{}'.format(self.shell_script_name)), 'w+') as f:
        
            for i in self.my_new_list:
                f.write(i)  
        os.chmod(os.path.join(self.base_path, '{}'.format(self.shell_script_name)), 0o777)


# In[ ]:


#chat = LEGO_shell_script(a,c,d,e)


# In[ ]:


#chat.shell_script()


# In[4]:


my_list_para = []
for i in c:
    #print(i)
    FileHandler = open(os.path.join(a,'lego_{}.input'.format(i)),"r")
    my_list_para.append(FileHandler.readlines())
    for j in my_list_para[-1]:
        if j.split()[0] == 'outfile':
            print(j)
    #print(my_list_para[0])
    #my_list_para = my_list_para[0]


# In[ ]:


my_list_para
# for i in c:
#     print(i)


# In[4]:


##### creates a .xyz file from the IMD file in same name as the IMD file. the created .xyz file is in the 
###   same format as mentioned in the link : https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/xyz.html 


class xyzgenerator:  
    def __init__(self, file_path, element):
    #def __init__(self):
        
#         self.remote_obj = Remote_Control()
        
#         self.base_path, self.element, self.parameter, self.IMD_file_name = self.remote_obj.input_func()

        self.file_path = file_path
        
        #self.file_path = os.path.join(self.base_path,self.IMD_file_name)  ##path of the IMD file
        
        self.filename = self.file_path.split('/')[-1].split('.')[0]+'.'+'xyz'  ##creating the name for the .xyz file from the path
        
        #self.filename = self.IMD_file_name.split('.')[0]+'.'+'xyz'  ##creating the name for the .xyz file from the IMD file name
        
        self.base_path = "/".join(self.file_path.split('/')[0:-1])  ## main base path where the param files, IMD files are kept and the .xyz file will be created
        
        self.element = element  ##symbol of the element used
        
        self.count = 0              ##saves the number of lines the header has except the last header line '#E\n'
        
        self.my_list = []           ##contains all the content of the IMD file after reading it
        
        self.new_list = []          ##contains all the lines of the IMD file expect the header part
        
        self.my_new_list =[]     ##contains only the x,y,z coordinates and along with the total number of entries at the top 
                                 ##replicating the .xyz format
            
        self.flag = True         ##this boolean indicates whether there are any columns for atom velocities and EPot or not
    
    def generator(self):
        
        FileHandler = open(self.file_path,"r")
        self.my_list.append(FileHandler.readlines())
        self.my_list = self.my_list[0]
        
        if int(self.my_list[0].split()[-1])==0 and int(self.my_list[0].split()[-2])==0:
            print('there is no columns for the atom velocities and no columns for the atom data, e.g. Epot')
            self.flag = True
    
            for i in range(0,len(self.my_list)):
            
                if self.my_list[i] == '#E\n' or self.my_list[i] == '#E \n' or self.my_list[i] == '#E':
                
                    break
                else:
                
                    self.count = self.count+1
                #self.new_list.append(self.my_list[i])
        else:
            print('there are columns for the atom velocities or columns for the atom data, e.g. Epot or both')
            self.flag = False
    
        self.new_list = self.my_list[self.count+1 :]
        
        
        #self.element = input("Element used : ")
        #print(self.element)
        
        #my_new_new_list = []
        while self.new_list[0] == '\n' or self.new_list[0] == '#\n':
            
            self.new_list.pop(0)
        
        if isinstance(int(self.new_list[0].split()[0]), int) == True:
            
            for i in range(0, len(self.new_list)):
                
                if self.flag == True:
                    
                    self.my_new_list.append(self.element+" "+" ".join(self.new_list[i].split()[-3 :])+" "+'\n')
            
            #my_new_new_list.append
            #" ".join(my_new_new_list[0])
        
        self.my_new_list.insert(0, str(len(self.new_list))+" "+'\n')
        
        self.my_new_list.insert(1, '##'+" "+'\n')
        
        return self.my_new_list
    
    def save_xyz(self):
        
        #filename = self.file_path.split('/')[-1].split('.')[0]+'.'+'xyz'
        
        list_temp = self.generator()
        
        #with open(os.path.join("/".join(self.file_path.split('/')[0:-1]), '{}'.format(filename)), 'w') as f:
        with open(os.path.join(self.base_path, '{}'.format(self.filename)), 'w') as f:
            #print(f)
            
            for i in list_temp:
            
                f.write(i)


# In[ ]:


a = xyzgenerator()


# In[ ]:


a.generator()


# In[5]:


##### creates a EXTENDED xyz file from the IMD file in same name_extended as the IMD file. the created .xyz file is in the 
###   same format as mentioned in the link : https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/xyz.html. This class also 
##   preserves the Boundary coordinates mentioned in the parameter files and also preserves the atom type property by making 
##   an additional column in the generated extended xyz file.


class xyzgenerator_extended:  
    def __init__(self, file_path, element, parameter_list):
        
        self.file_path = file_path  ##path of the IMD file
        
        self.parameter_list = parameter_list
        
        self.filename = self.file_path.split('/')[-1].split('.')[0]+'_'+'extended'+'.'+'extxyz'  ##creating the name for the .xyz file from the path
        
        self.base_path = "/".join(self.file_path.split('/')[0:-1])## main base path where the param files, IMD files are kept and the .xyz file will be created
        
        #self.lego_input_path = os.path.join(self.base_path, '{}'.format('lego.input'))
        self.lego_input_path = os.path.join(self.base_path, 'lego_{}.input'.format(self.parameter_list[0]))
        
        self.element = element  ##symbol of the element used
        
        self.count = 0              ##saves the number of lines the header has except the last header line '#E\n'
        
        self.my_list = []           ##contains all the content of the IMD file after reading it
        
        self.new_list = []          ##contains all the lines of the IMD file expect the header part
        
        self.my_new_list =[]     ##contains only the x,y,z coordinates and along with the total number of entries at the top 
                                 ##replicating the .xyz format
            
        self.input_list = []
        
        #self.Boundary_coordinates = []
            
        self.flag = True  ##this boolean indicates whether there are any columns for atom velocities and EPot or not
    
    def read_LEGO_input(self):
        
        Boundary_coordinates = []
        
        FileHandler = open(self.lego_input_path,"r")
        self.input_list.append(FileHandler.readlines())
        #self.input_list = self.input_list[0]
        
        for i in self.input_list[-1]:
            if i.split()[0] == 'box_x':
                Boundary_coordinates.append(i.split()[1])
        
            if i.split()[0] == 'box_y':
                Boundary_coordinates.append(i.split()[1])
        
            if i.split()[0] == 'box_z':
                Boundary_coordinates.append(i.split()[1])
                
        if len(Boundary_coordinates)<3:
            
            Boundary_coordinates.clear()
            Boundary_coordinates.append('5.44')
            Boundary_coordinates.append('5.44')
            Boundary_coordinates.append('5.44')
            
            warnings.warn('All boundary box coordinates are unavailable. will take the default value', UserWarning)
        
        return Boundary_coordinates
    
    def generator_extd(self):
        
        FileHandler = open(self.file_path,"r")
        self.my_list.append(FileHandler.readlines())
        self.my_list = self.my_list[0]
        
        if int(self.my_list[0].split()[-1])==0 and int(self.my_list[0].split()[-2])==0:
            print('there is no columns for the atom velocities and no columns for the atom data, e.g. Epot')
            self.flag = True
    
            for i in range(0,len(self.my_list)):
            
                if self.my_list[i] == '#E\n' or self.my_list[i] == '#E \n' or self.my_list[i] == '#E':
                
                    break
                else:
                
                    self.count = self.count+1
                #self.new_list.append(self.my_list[i])
        else:
            print('there are columns for the atom velocities or columns for the atom data, e.g. Epot or both')
            self.flag = False
    
        self.new_list = self.my_list[self.count+1 :]
        
        
        #self.element = input("Element used : ")
        #print(self.element)
        
        #my_new_new_list = []
        while self.new_list[0] == '\n' or self.new_list[0] == '#\n':
            
            self.new_list.pop(0)
        
        if isinstance(int(self.new_list[0].split()[0]), int) == True:
            
            for i in range(0, len(self.new_list)):
                
                if self.flag == True:
                    
                    #self.my_new_list.append(self.element+" "+" ".join(self.new_list[i].split()[-3 :])+" "+ self.new_list[i].split()[1]+" "+'\n')
                    #self.my_new_list.append(self.new_list[i].split()[2]+" "+" ".join(self.new_list[i].split()[-3 :])+" "+ self.new_list[i].split()[1]+" "+'\n')
                    self.my_new_list.append(self.element+" "+" ".join(self.new_list[i].split()[-3 :])+" "+ self.new_list[i].split()[1]+" "+'\n')
            #my_new_new_list.append
            #" ".join(my_new_new_list[0])
        
        self.my_new_list.insert(0, str(len(self.new_list))+" "+'\n')
        
        Boundary_coordinates = self.read_LEGO_input()
        
        #self.my_new_list.insert(1, '##'+" "+'\n')
        #self.my_new_list.insert(1, 'Lattice="5.44 0.0 0.0 0.0 5.44 0.0 0.0 0.0 5.44"'+" "+'\n')
        #self.my_new_list.insert(1, 'Lattice="H11 H21 H31 H12 H22 H32 H13 H23 H33"'+" "+'Properties=molecule_type:S:1:pos:R:3:atom_types:I:1 Time=0.0'+" "+'\n')
        self.my_new_list.insert(1, 'Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"'.format(Boundary_coordinates[0],Boundary_coordinates[1],Boundary_coordinates[2])+" "+'Properties=symbols:S:1:pos:R:3:tag:I:1'+" "+'\n')
        
        #self.my_new_list.insert(2, 'Properties=species:S:1:pos:R:3:atom_types:I:1'+" "+'\n')
        
        
        return self.my_new_list
    
    def save_xyz(self):
        
        #filename = self.file_path.split('/')[-1].split('.')[0]+'.'+'xyz'
        
        list_temp = self.generator_extd()
        
        #with open(os.path.join("/".join(self.file_path.split('/')[0:-1]), '{}'.format(filename)), 'w') as f:
        with open(os.path.join(self.base_path, '{}'.format(self.filename)), 'w') as f:
            #print(f)
            
            for i in list_temp:
            
                f.write(i)


# In[ ]:


a = xyzgenerator()


# In[6]:


class LEGO(TemplateJob, HasStructure):
    def __init__(self, project=None, job_name=None):
        super().__init__(project=project, job_name=job_name)
        # self.input is predefined by the TemplateJob and will be stored automatically
        self.input.update({
            # add the default values from ex.conf parameter file here
            'structure': 'fcc',
            'filename': 'output.xyz',
        })
        #self.input = {}
        # placeholder shell script, modify the script to point to your local LEGO installation
        # you might need to update this to the absolute path where you put the lego.sh file
        # later on this shell script will be provided via the pyiron_resources and this line 
        # will then be unnecessary
        #self.working_directory = os.path.abspath(r'/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION/')
        #self.executable = os.path.join(os.getcwd(), 'lego.sh')
        #a = xyzgenerator_extended()
        #self.path = path
        
        
        
        self.remote_obj = Remote_Control()
        self.base_path, self.element, self.parameter_list, self.header_file, self.IMD_file_name = self.remote_obj.input_func()

        self.input_list = []
        self.mylist = []
        self.read_input()
        #self.write_input()
        #self.executable = os.path.join(r'/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION', 'lego.sh')
        self.shell_obj = LEGO_shell_script(self.base_path, self.parameter_list, self.header_file, self.IMD_file_name)
        self.shell_obj.shell_script()

#         self.shell_script()
        
#         self.shell_script_name = 'lego'+'_'+'{}'.format(self.IMD_file_name.split('.')[0])+'.'+'sh'
        
        #self.executable = os.path.join(self.base_path, 'lego.sh')
        self.executable = os.path.join(self.base_path, '{}'.format(self.shell_obj.shell_script_name))
        #self.executable = os.path.join(self.base_path, '{}'.format(self.shell_script_name))
        #self.mylist = []
        self.para = []
        #self.a = 'a'
        #self.my_new_list = []
        
        #self.mydict = {}
        #self.read_input()
#         self.write_input()
        #self.collect_output()
#         self._number_of_structures()
        #self._get_structure()
        #self.output_path = ''
    # convenience function for users with old parameter files#
#     def input_parameters(self):
#         para = input("Enter LEGO parameter's path:")
#         self.para.append(para)
#         return self.para[0]

#     def input_func(self):
        
#         path = input('Enter the base path : ')
#         print('the base path is : {}'.format(path))
#         element = input('Enter Element name :')
#         print('the element name is : {}'.format(element))
#         parameter = input ('Enter the parameter file name : ')
#         print('the element name is : {}'.format(parameter))
#         IMD_file_name = input('Enter the IMD file name : ')
#         print('The IMD file name is : {}'.format(IMD_file_name))
        
#         return path,element,parameter,IMD_file_name
    
   

    # def shell_script(self):
# #         FileHandler = open(self.path,"r")
# #         self.my_list.append(FileHandler.readlines())
# #         self.my_list = self.my_list[0]
#         self.my_new_list.extend(['#!/bin/bash\n','\n','current_path="{}"\n'.format(self.base_path),'present_path=$(pwd)\n',
#                                  '\n','cd $current_path\n','\n','$current_path/legoV3.1 $current_path/{}\n'.format(self.input_param),'\n',
#                                  '#mv \n','\n','cp $current_path/header.txt $current_path/{}\n'.format(self.IMD_file_name),'\n',
#                                  'cat 111.fcc >> {}\n'.format(self.IMD_file_name),'\n','chmod +rwx {}\n'.format(self.IMD_file_name),'\n'])
       
#         with open(os.path.join(self.base_path, '{}'.format(self.shell_script_name)), 'w') as f:
        
#             for i in self.my_new_list:
#                 f.write(i)  
                
    #def read_input(self, filename = r'/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION/ex.confparam'):
    def read_input(self):
        for i in range(0,len(self.parameter_list)):
            filename_ = os.path.join('{}'.format(self.base_path), '{}'.format(self.parameter_list[i]))
            self.a = self.parameter_list[i]
            temp_list = []
            values = []
            keys = []
        #myDict = {}
        #with open(os.path.join(r'/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION', 'ex.confparam'), 'r') as f:
            with open(filename_) as f:
                for a in f:
                    temp_list.append(a)
                    #print(os.getcwd())
                    #print(temp_list)
                    if temp_list[0]== '\n' or list(temp_list[0])[0] == '#':
                        temp_list = []
                    else:
                        self.mylist.append(a)
                        temp_list = []
        
            for i in self.mylist:
                #print(i.split())
                #myDict[i.split()[0]] = i.split()[1]
                keys.append(i.split()[0])
                values.append(i.split()[1:])
    
            for i in range(0,len(keys)):
            
                self.input[keys[i]] = ' '.join(values[i])
                
            self.input_list.append(self.input)
            #self.a = self.parameter_list[i]
            self.write_input()
        
#         if self.status.finished:
#             return 1
#         else:
#             return 0
        #self.input
            
            # this should read in the key value pairs from the given file and put them into self.input
    
    # necessary overload for TemplateJob
    def write_input(self):
        filename_input = self.a
        #with open(os.path.join(r'/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION', 'lego.input'), 'w') as f:
        #for i in range(0,len(self.parameter_list)):
        with open(os.path.join(self.base_path, 'lego_{}.input'.format(filename_input)), 'w') as f:
            for k, v in self.input.items():
                    # check with the lego input format whether this is correct
                    f.write(f'{k} {v}\n')
                
   
    def IMD_file_creation(self):
        
        IMD_file_creation_list = []
        for i in self.parameter_list:
            FileHandler = open(os.path.join(self.base_path,'{}'.format(i)),"r")
            IMD_file_creation_list.append(FileHandler.readlines())
            IMD_file_creation_list = IMD_file_creation_list[0]

    # necessary overload for TemplateJob
    def collect_output(self):
        #output_path = os.path.join(self.working_directory, self.input.outfile)
        #output_path = os.path.join(r'/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION', '{}'.format(self.input.filename))
        #generator  = xyzgenerator_extended(r'/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION/dynamic_edgedislo_bv_1-10_lv_11-2_Ni_x1-10_y11-2_z111.fcc', 'Au')
        generator  = xyzgenerator_extended(os.path.join('{}'.format(self.base_path), '{}'.format(self.IMD_file_name)), self.element, self.parameter_list)
        print(generator.base_path, generator.lego_input_path)
        generator.save_xyz()
        print(generator.input_list)
        output_path = os.path.join(generator.base_path, '{}'.format(generator.filename))
        print(output_path)
        self.output.structure = self.project.create.structure.read(output_path)
        self.to_hdf()
        
    # necessary overload for HasStructure
    def _number_of_structures(self):
        if self.status.finished:
            return 1
        else:
            return 0
        
    # necessary overload for HasStructure
    def _get_structure(self, frame=-1, wrap_atoms=True):
        return self.output.structure


# In[ ]:





# In[27]:


# class LEGO(TemplateJob, HasStructure):
#     def __init__(self, project=None, job_name=None):
#         super().__init__(project=project, job_name=job_name)
#         # self.input is predefined by the TemplateJob and will be stored automatically
#         self.input.update({
#             # add the default values from ex.conf parameter file here
#             'structure': 'fcc',
#             'filename': 'output.xyz',
#         })
#         #self.input = {}
#         # placeholder shell script, modify the script to point to your local LEGO installation
#         # you might need to update this to the absolute path where you put the lego.sh file
#         # later on this shell script will be provided via the pyiron_resources and this line 
#         # will then be unnecessary
#         #self.working_directory = os.path.abspath(r'/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION/')
#         #self.executable = os.path.join(os.getcwd(), 'lego.sh')
#         #a = xyzgenerator_extended()
#         #self.path = path
        
#         self.parameter_list = []
        
#         self.base_path = input('Enter the base path : ')
#         print('the base path is : {}'.format(self.base_path))
#         self.element = input('Enter Element name :')
#         print('the element name is : {}'.format(self.element))
#         warnings.warn('If there are more than 1 paramter file then please put the names of all the files in the below field seperated by comma', UserWarning)
#         self.parameter = input ('Enter the parameter file name : ')
#         print('the parameter name is : {}'.format(self.parameter))
#         self.header_file = input('Enter the header file name : ' )
#         print('the header file name is : {}'.format(self.header_file))
#         self.IMD_file_name = input('Enter the IMD file name : ')
#         print('The IMD file name is : {}'.format(self.IMD_file_name))
        
#         for i in range(0,len(self.parameter.split(','))):
#             self.parameter_list.append(self.parameter.split(',')[i].split()[0])
        
#         #self.remote_obj = Remote_Control()
#         #self.base_path, self.element, self.parameter_list, self.header_file, self.IMD_file_name = self.input_func()

#         self.input_list = []
#         self.mylist = []
#         self.read_input()
#         #self.write_input()
#         #self.executable = os.path.join(r'/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION', 'lego.sh')
#         self.shell_obj = LEGO_shell_script(self.base_path, self.parameter_list, self.header_file, self.IMD_file_name)
#         self.shell_obj.shell_script()

# #         self.shell_script()
        
# #         self.shell_script_name = 'lego'+'_'+'{}'.format(self.IMD_file_name.split('.')[0])+'.'+'sh'
        
#         #self.executable = os.path.join(self.base_path, 'lego.sh')
#         self.executable = os.path.join(self.base_path, '{}'.format(self.shell_obj.shell_script_name))
#         #self.executable = os.path.join(self.base_path, '{}'.format(self.shell_script_name))
#         #self.mylist = []
#         self.para = []
                
#     #def read_input(self, filename = r'/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION/ex.confparam'):
    
# #     def input_func(self):
        
# #         parameter_list = []
        
# #         path = input('Enter the base path : ')
# #         print('the base path is : {}'.format(path))
# #         element = input('Enter Element name :')
# #         print('the element name is : {}'.format(element))
# #         warnings.warn('If there are more than 1 paramter file then please put the names of all the files in the below field seperated by comma', UserWarning)
# #         parameter = input ('Enter the parameter file name : ')
# #         print('the parameter name is : {}'.format(parameter))
# #         header_file = input('Enter the header file name : ' )
# #         print('the header file name is : {}'.format(header_file))
# #         IMD_file_name = input('Enter the IMD file name : ')
# #         print('The IMD file name is : {}'.format(IMD_file_name))
        
# #         for i in range(0,len(parameter.split(','))):
# #             parameter_list.append(parameter.split(',')[i].split()[0])
# #         #print(b)
        
# #         #return path,element,parameter,IMD_file_name,parameter_list
# #         return path,element,parameter_list,header_file,IMD_file_name
    
#     def read_input(self):
#         #self.base_path, self.element, self.parameter_list, self.header_file, self.IMD_file_name = self.input_func()
#         for i in range(0,len(self.parameter_list)):
#             filename_ = os.path.join('{}'.format(self.base_path), '{}'.format(self.parameter_list[i]))
#             self.a = self.parameter_list[i]
#             temp_list = []
#             values = []
#             keys = []
#         #myDict = {}
#         #with open(os.path.join(r'/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION', 'ex.confparam'), 'r') as f:
#             with open(filename_) as f:
#                 for a in f:
#                     temp_list.append(a)
#                     #print(os.getcwd())
#                     #print(temp_list)
#                     if temp_list[0]== '\n' or list(temp_list[0])[0] == '#':
#                         temp_list = []
#                     else:
#                         self.mylist.append(a)
#                         temp_list = []
        
#             for i in self.mylist:
#                 #print(i.split())
#                 #myDict[i.split()[0]] = i.split()[1]
#                 keys.append(i.split()[0])
#                 values.append(i.split()[1:])
    
#             for i in range(0,len(keys)):
            
#                 self.input[keys[i]] = ' '.join(values[i])
                
#             self.input_list.append(self.input)
#             #self.a = self.parameter_list[i]
#             self.write_input()
        
# #         if self.status.finished:
# #             return 1
# #         else:
# #             return 0
#         #self.input
            
#             # this should read in the key value pairs from the given file and put them into self.input
    
#     # necessary overload for TemplateJob
#     def write_input(self):
#         filename_input = self.a
#         #with open(os.path.join(r'/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION', 'lego.input'), 'w') as f:
#         #for i in range(0,len(self.parameter_list)):
#         with open(os.path.join(self.base_path, 'lego_{}.input'.format(filename_input)), 'w') as f:
#             for k, v in self.input.items():
#                     # check with the lego input format whether this is correct
#                     f.write(f'{k} {v}\n')
                
   #     def IMD_file_creation(self):
        
#         IMD_file_creation_list = []
#         for i in self.parameter_list:
#             FileHandler = open(os.path.join(self.base_path,'{}'.format(i)),"r")
#             IMD_file_creation_list.append(FileHandler.readlines())
#             IMD_file_creation_list = IMD_file_creation_list[0]

#     # necessary overload for TemplateJob
#     def collect_output(self):
#         #output_path = os.path.join(self.working_directory, self.input.outfile)
#         #output_path = os.path.join(r'/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION', '{}'.format(self.input.filename))
#         #generator  = xyzgenerator_extended(r'/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION/dynamic_edgedislo_bv_1-10_lv_11-2_Ni_x1-10_y11-2_z111.fcc', 'Au')
#         generator  = xyzgenerator_extended(os.path.join('{}'.format(self.base_path), '{}'.format(self.IMD_file_name)), self.element, self.parameter_list)
#         print(generator.base_path, generator.lego_input_path)
#         generator.save_xyz()
#         print(generator.input_list)
#         output_path = os.path.join(generator.base_path, '{}'.format(generator.filename))
#         print(output_path)
#         self.output.structure = self.project.create.structure.read(output_path)
#         self.to_hdf()
        
#     # necessary overload for HasStructure
#     def _number_of_structures(self):
#         if self.status.finished:
#             return 1
#         else:
#             return 0
        
#     # necessary overload for HasStructure
#     def _get_structure(self, frame=-1, wrap_atoms=True):
#         return self.output.structure


# In[9]:


pr = Project('test300')
pr.remove_jobs_silently(recursive=True)
job = pr.create_job(job_type=LEGO, job_name="LEGOjob", delete_existing_job=True)


# In[ ]:


job.input_list


# In[ ]:


a = []
a.append(job.input)
for k,v in a[0].items():
    print(k,v)


# In[10]:


job.run()


# In[ ]:


job.xyz


# In[11]:


job.get_structure().plot3d()


# In[ ]:


a = xyzgenerator_extended(r'/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION/dynamic_edgedislo_bv_1-10_lv_11-2_Ni_x1-10_y11-2_z111.fcc', 'Au')


# In[ ]:


a.generator_extd()


# In[ ]:


a.my_list


# In[ ]:


b = a.new_list


# In[ ]:


b[0].split()[-3 :]


# In[ ]:


b[0].split()


# In[ ]:


a.element+" "+" ".join(a.new_list[0].split()[-3 :])+" "+a.new_list[0].split()[1]


# In[ ]:


a.elementment


# In[ ]:


b[0].split()[1]


# In[ ]:


c = xyzgenerator(r'/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION/dynamic_edgedislo_bv_1-10_lv_11-2_Ni_x1-10_y11-2_z111.fcc', 'Au')


# In[ ]:


c.save_xyz()


# In[ ]:


a = xyzgenerator_extended(r'/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION/dynamic_edgedislo_bv_1-10_lv_11-2_Ni_x1-10_y11-2_z111.fcc', 'Au')


# In[ ]:


a.read_LEGO_input()


# In[ ]:


a.input_list


# In[ ]:


yo = []
for i in a.input_list:
    if i.split()[0] == 'box_x':
        yo.append(i.split()[1])
        
    if i.split()[0] == 'box_y':
        yo.append(i.split()[1])
        
    if i.split()[0] == 'box_z':
        yo.append(i.split()[1])
    
    
print(yo)
yo.pop(2)

if len(yo)<3 :
    raise Warning('All boundary box coordinates are unavailable. will take the default value')
    
    


# In[ ]:


os.path.join(a.base_path, '{}'.format('lego.input'))


# In[ ]:


b = LEGO()


# In[ ]:


job.input


# In[ ]:


a = ['yo', 2 , 3]


# In[ ]:


a.clear()


# In[ ]:


a


# In[ ]:


class LEGO_shell_script:
    
    def __init__(self,base_path,input_param, IMD_file_name):
        self.base_path = base_path
        self.input_param = input_param
        self.IMD_file_name = IMD_file_name
        self.shell_script_name = 'lego'+'_'+'{}'.format(self.IMD_file_name.split('.')[0])+'.'+'sh'
        #self.my_list = []
        self.my_new_list = []
        
    def shell_script(self):
#         FileHandler = open(self.path,"r")
#         self.my_list.append(FileHandler.readlines())
#         self.my_list = self.my_list[0]
        self.my_new_list.extend(['#!/bin/bash\n','\n','current_path="{}"\n'.format(self.base_path),'present_path=$(pwd)\n',
                                 '\n','cd $current_path\n','\n','$current_path/legoV3.1 $current_path/{}\n'.format(self.input_param),'\n',
                                 '#mv \n','\n','cp $current_path/header.txt $current_path/{}\n'.format(self.IMD_file_name),'\n',
                                 'cat 111.fcc >> {}\n'.format(self.IMD_file_name),'\n'])
       
        with open(os.path.join(self.base_path, '{}'.format(self.shell_script_name)), 'w') as f:
        
            for i in self.my_new_list:
                f.write(i)     


# In[ ]:


a = LEGO_shell_script('/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION', 'ex.confparam', 'Ni_x101_z1-1-1.fcc')


# In[ ]:


a.shell_script()


# In[ ]:


a.my_list


# In[ ]:


a.my_new_list[2]


# In[ ]:


'lego'+'_'+'{}'.format('yo')+'.'+'sh'


# In[ ]:


a.IMD_file_name.split('.')[0]


# In[ ]:


a.shell_script_name


# In[ ]:


a = input('your name : ')


# In[ ]:


len(a.split(','))


# In[ ]:


b = []


# In[ ]:


type(b)


# In[ ]:


for i in range(0,len(a.split(','))):
    b.append(a.split(',')[i].split()[0])
print(b)


# In[ ]:


yo = []
yo.extend(['#!/bin/bash\n','\n','current_path="{}"\n'.format('debu'),'present_path=$(pwd)\n',
                                 '\n','cd $current_path\n','\n','$current_path/legoV3.1 $current_path/{}\n'.format('debu'),'\n',
                                 '#mv \n','\n','cp $current_path/header.txt $current_path/{}\n'.format('debu'),'\n',
                                 'cat 111.fcc >> {}\n'.format('debu'),'\n'])


# In[ ]:


yo.append('cat 100.fcc >> debu\n')


# In[ ]:


yo


# In[ ]:


import datetime as dt
d = []
base_path = '/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION'
for i in os.listdir(base_path):
    
    LEGO_output_file_creation_time = dt.datetime.fromtimestamp(os.path.getctime(os.path.join(base_path,'{}'.format(i)))).timestamp()
    #LEGO_output_file_creation_time = LEGO_output_file_creation_time.split(' ')[-1].split(':')
    current_time = dt.datetime.now().timestamp()
    #current_time = current_time.split(' ')[-1].split(':')
    
    if i.split('.')[-1] == 'fcc' and int(current_time)-int(LEGO_output_file_creation_time) < 120:
        d.append(i)
    


# In[ ]:


os.listdir('/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION')[0].split('.')


# In[ ]:


d


# In[ ]:


os.path.getctime('/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION/111.fcc')


# In[ ]:


import datetime as dt

a  = dt.datetime.fromtimestamp(os.path.getctime('/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION/111.fcc')).strftime('%Y-%m-%d %H:%M:%S')


# In[ ]:


a = a.timestamp()


# In[ ]:


b = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# In[ ]:


b = b.split(' ')[-1].split(':')[1]


# In[ ]:


if int(b) - int(a) >2:
    print('yo')


# In[ ]:


type(a)


# In[ ]:


a


# In[ ]:


dt.datetime.fromtimestamp(os.path.getctime('/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION/lego_ex.confparam.input')).strftime('%Y-%m-%d %H:%M:%S')


# In[ ]:


dt.datetime.fromtimestamp(os.path.getctime('/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION/lego_Ni_IMD_new_deb.sh')).strftime('%Y-%m-%d %H:%M:%S')


# In[ ]:


dt.datetime.now().timestamp() - dt.datetime.fromtimestamp(os.path.getctime('/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION/debu.fcc')).timestamp()


# In[ ]:


para = ['debu.conf','chat.conf']
baby = []
for i in range (0,len(para)):
            baby.append('$current_path/legoV3.1 $current_path/{}\n'.format(para[i]))
            baby.append('\n')


# In[ ]:


baby


# In[ ]:


IMD_file_creation_list = []
FileHandler = open(os.path.join('/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION','lego_ex.confparam.input'),"r")
IMD_file_creation_list.append(FileHandler.readlines())
IMD_file_creation_list = IMD_file_creation_list[0]


# In[ ]:


para = []
for i in IMD_file_creation_list:
    if i.split()[0] == 'outfile':
        FileHandler = open(os.path.join('/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION','Ni_IMD_new.fcc'),"r")
        IMD_file_creation_list.append(FileHandler.readlines())
        IMD_file_creation_list = IMD_file_creation_list[0]
        para.append(i.split()[-1])


# In[ ]:


para =para
for i in IMD_file_creation_list:
    if i.split()[0] == 'outfile':
        para = i.split()[-1]


# In[ ]:


para.split('.')


# In[ ]:


input_list = []
for i in os.listdir('/root/LEGO/1_CRYSTAL_GENERATION/1_CRYSTAL_GENERATION/'):
    if i == 'lego.input':
        print(True)


# In[ ]:


a = {'a' : 2,
    'b' : 40}


# In[ ]:


a


# In[ ]:


b = {'a' : 30, 'b': 23}


# In[ ]:


b


# In[ ]:


goo = []
goo.append(a)
goo.append(b)


# In[ ]:


goo


# In[ ]:


goo[1]


# In[ ]:


for i in range(0,2):
    print(i)


# In[18]:


Boundary_coordinates = []
input_list = []

FileHandler = open('/root/LEGO/2_TWIN-BOUNDARY/2_TWIN-BOUNDARY/test/lego_lower_crystal.confparam.input',"r")
input_list.append(FileHandler.readlines())
#input_list = input_list[0]

for i in input_list[-1]:
    if i.split()[0] == 'box_x':
        Boundary_coordinates.append(i.split()[1])

    if i.split()[0] == 'box_y':
        Boundary_coordinates.append(i.split()[1])

    if i.split()[0] == 'box_z':
        Boundary_coordinates.append(i.split()[1])


# In[19]:


Boundary_coordinates


# In[16]:


input_list[-1]


# In[22]:


a = xyzgenerator_extended('/root/LEGO/2_TWIN-BOUNDARY/2_TWIN-BOUNDARY/test/IMD_Ni.fcc','Ni',['lego_upper_crystal.confparam.input', 'lego_lower_crystal.confparam.input'])


# In[23]:


a.save_xyz()


# In[24]:


a.file_path


# In[26]:


a.lego_input_path


# In[ ]:


a.input_list[0].split()


# In[ ]:




