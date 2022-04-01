variable "instance_name" {
  description = "Value of the Name tag for the EC2 instance"
  type        = string
  default     = "SeaFlowPSDSpotInstance"
}

variable "security_group" {
  description = "Security group ID for the EC2 instance"
  type        = string
  default     = "ssh"
}

variable "instance_ami" {
  description = "AMI for the EC2 instance"
  type        = string
  default     = "ami-0892d3c7ee96c0bf7"
}

variable "instance_type" {
  description = "AMI for the EC2 instance"
  type        = string
  default     = "c6a.16xlarge"
}
