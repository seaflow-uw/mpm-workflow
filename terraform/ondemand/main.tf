terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 3.27"
    }
  }

  required_version = ">= 0.14.9"
}

provider "aws" {
  profile = "default"
  region  = "us-west-2"
}

# resource "aws_key_pair" "ubuntu" {
#   key_name   = "ubuntu"
#   public_key = file("key.pub")
# }

# resource "aws_security_group" "ubuntu" {
#   name        = "ubuntu-security-group"
#   description = "Allow SSH traffic"

#   ingress {
#     description = "SSH"
#     from_port   = 22
#     to_port     = 22
#     protocol    = "tcp"
#     cidr_blocks = ["0.0.0.0/0"]
#   }

#   egress {
#     from_port   = 0
#     to_port     = 0
#     protocol    = "-1"
#     cidr_blocks = ["0.0.0.0/0"]
#   }

#   tags = {
#     Name = "terraform"
#   }
# }

resource "aws_instance" "seaflow_psd_server" {
  key_name        = "chris-aws"
  ebs_optimized   = true
  ami             = var.instance_ami
  instance_type   = var.instance_type
  security_groups = [var.security_group]

  root_block_device {
    volume_size = 30
  }

  tags = {
    Name = var.instance_name
  }
}
