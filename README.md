# Dropout Neural Networks and Kalman Filtering for Robust online airstrip surface estimation for optimized braking 


## License
Please be aware that this code was originally implemented for research purposes and may be subject to changes and any fitness for a particular purpose is disclaimed. 
```
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 ```

## Acknowledgement
This work was funded by Clean Sky 2 Joint Undertaking (CS2JU) under 
the European Union’s Horizon 2020 research and innovation programme, under project no. 821079,
E-Brake.
For more details visit the link: <https://www.maregroup.it/cleansky2/?lang=en>

### Disclaimer: the repository is work in progress
That means we are working on uploading and upgrading all the functionalities.


## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### CUDA 10.1 ###
Cuda v. 10.1 required, visit https://developer.nvidia.com/cuda-downloads


### Setting up the virtual Environment ###
Move into the installation folder and execute the installation script:  
```shell
./01_InstallAnacondaEnv.sh
```
It will install Anaconda and will set-up the virtual environment and all the packages needed.   
The name of the environment will be 'Ebrake' and it's carried with python 3.6.5. For more information about the installed packages, open the requirements.txt file.  
For more information about the Anaconda3 usage, please visit <https://docs.conda.io/projects/conda/en/latest/commands.html>  


### Matlab Compatibilty###
Tested configurations: 

-Ubuntu 16.04 , Python 3.6.5 , Matlab R2018a

-Ubuntu 18.04 , Python 3.6.5 , Matlab R2018b

-Ubuntu 20.04, Python 3.6.5, matlab R2020a or b

### Setting up the python interpeter in Matlab###
For more information please visit:    
* pyenv guide <https://www.mathworks.com/help/matlab/ref/pyenv.html> (matlab >= r2020a)  
* pyversion guide <https://it.mathworks.com/help/matlab/ref/pyversion.html> (matlab < r2020a)  
#### Matlab2020 ####
  pe = pyenv('version',executable_path) : specifies the full path to the Python executable. You can use this syntax on any platform or for repackaged CPython implementation downloads.  
  example: pyenv('Version', '/home/pluto/anaconda3/envs/Ebrake/bin/python')
  
#### Matlab2018 ####
  ___ = pyversion executable specifies the full path to the Python executable. You can use this syntax on any platform or for repackaged CPython implementation downloads.  


### Troubleshooting ###
When python is running within MATLAB, it ends up using MATLAB's MKL.  
It looks like your Python code is incompatible with MATLAB's MKL, probably due to incompatible compile-time options.  
In matlab script before launch anything

flag = int32(bitor(2, 8));

py.sys.setdlopenflags(flag);


### Troubleshooting 16.04 MATLAB freezed ###
1- Edit following file:
nano /etc/default/grub
2- Edit the variable GRUB_CMDLINE_LINUX_DEFAULT to only have the content "nomodeset"
3- Run on terminal: sudo grub-mkconfig -o /boot/grub/grub.cfg
4- Reboot.



