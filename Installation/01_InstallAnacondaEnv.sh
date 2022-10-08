#!/bin/bash
set -e
shopt -s extglob
shopt -s globstar

 # Author: Francesco Crocetti (francesco.crocetti@unipg.it)
 # Description: Installation Anaconda Env

 
 hello() {
    echo -e " _____  _____         _____  _               ____    
|_   _|/ ____|  /\   |  __ \| |        /\   |  _ \  
  | | | (___   /  \  | |__) | |       /  \  | |_) | 
  | |  \___ \ / /\ \ |  _  /| |      / /\ \ |  _ <  
 _| |_ ____) / ____ \| | \ \| |____ / ____ \| |_) |
|_____|_____/_/    \_\_|  \_\______/_/    \_\____/ "

  echo -e "The installation will install Anaconda3 and setup the virtual environment for further applications\n"
}

global_path=${PWD}
env_name=Ebrake
python_ver=3.6.5


check_installation() {
	trap 'catch $? $LINENO' EXIT # Installs trap now
	catch() {
	  if [ "$1" != "0" ]; then
	    echo "An error occured during the installation of Ananconda." >&2
	  fi
	}
}


install_anaconda() {

	 echo -e "\n -->Checking conda installation; \n"
	 if ! which conda>/dev/null; then
        read -r -p "Command $command not found, install it? [y/N] " REPLY
        case "$REPLY" in
          [yY][eE][sS]|[yY])
            anaconda_version=Anaconda3-2020.02-Linux-x86_64.sh
            wget https://repo.anaconda.com/archive/$anaconda_version
            chmod +x $anaconda_version
            ./$anaconda_version
            echo "Removing installation file"
            rm -v $anaconda_version
            echo "Conda installation Done"
            ;;
          *)
            echo "Exiting..."
            exit
            ;;
        esac
	 else
	  conda_command=$(which conda) 
	  echo "Ok! installation found in: >>> $conda_command <<<"
	  read -r -p " root path of Anacaonda3:$(dirname "$(dirname "$conda_command")"); is the path right? [y/N] " REPLY 
	  case "$REPLY" in
	    [yY][eE][sS]|[yY])
		    ;;
	    *)
        read -r -p "Insert new path" newRoot
        $root_path_anaconda = $newRoot
        echo "root path for anaconda3: >>> $root_path_anaconda <<<"
        ;;
	  esac
	fi
 }
 
install_anaconda_env(){
	echo -e "-->Creating virtual environment $env_name" 
	conda create -n $env_name python=$python_ver
	read -r -p "is the installation successfully completed? [y/N] " REPLY 
	case "$REPLY" in
	   [yY][eE][sS]|[yY]) 
	   	. "$HOME/.bashrc" 
	   	. ~/anaconda3/etc/profile.d/conda.sh 
		conda activate $env_name 
		;;
	   *)
		echo "Exiting..."
	 exit
		;;
	esac 		  
}
 

activate_anaconda_env() {
	echo -e "-->Trying to activate the environment: $env_name"
	. "$HOME/.bashrc" 
	source ~/anaconda3/etc/profile.d/conda.sh 
	conda activate $env_name 
	
	if [ $CONDA_DEFAULT_ENV == $env_name ]; then
		echo "Environment: $env_name successfully activated!"
	else
		echo "Failed to activate the environment: $env_name, is that correctly installed?"
		exit
	fi
	
}

install_dependencies(){
	echo -e "-->Dependencies Installation from requirements.txt by using Pip"
	conda install pip
	pip install -r requirements.txt


}

subcommand=$1
 
 case "$subcommand" in
    *)                         
    hello
    check_installation
    install_anaconda_env
    activate_anaconda_env
    install_dependencies
    ;;                                                                                 
esac 

