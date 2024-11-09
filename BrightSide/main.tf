terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      # version = "~> 4.16"
      version = "~> 5.0"
    }
  }

  required_version = ">= 1.2.0"
}

provider "aws" {
  region = "us-west-2"
  # region = "us-east-2"
}


resource "aws_instance" "app_server" {
  # provisioner "file" {
  #   source = "app.py"
  #   destination = "/home/ubuntu/app.py"
  # }
  ami           = "ami-08d70e59c07c61a3a" 
  #ami           = "ami-830c94e3"
  
  instance_type = "t2.micro"
  # subnet_id = module.vpc.public_subnets[0]

  tags = {
    Name = "HackUmassApp"
  }

  user_data = file("${path.module}/user_data.sh")
}

# module "ec2_instances" {
  
#   source  = "terraform-aws-modules/ec2-instance/aws"
#   version = "5.6.0"  # Check for the latest version

#   name           = "flask-app-instance"

#   ami                    = "ami-08d70e59c07c61a3a"  # Replace with the correct AMI for your region and OS
#   instance_type          = "t2.micro"      # Adjust the instance type based on your needs
#   subnet_id              = module.vpc.public_subnets[0]

#   tags = {
#     "Name"        = "FlaskAppInstance"
#     "Environment" = "Development"
#   }
#   user_data = file("${path.module}/user_data.sh")

# }