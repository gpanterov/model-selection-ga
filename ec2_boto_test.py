import boto
import time
import boto.manage.cmdshell
import os
import aws_tools as aws_tools

reload(aws_tools)
#ec2 = boto.connect_ec2()
#ec2 = boto.connect_ec2(aws_access_key_id='AKIAICBVHKUYHQSPWUAA1',
#	aws_secret_access_key='HLTjJI6HTd+BYI68UeYfD5hTSknkeZbmsgZhTeNh')

AMI = 'ami-6692700e'
key_name='aws_ubuntu_key'
user_name = 'ubuntu'
#instance, cmd = launch_instance(ami=AMI, key_name=key_name)
ec2 = boto.connect_ec2()
reservations = ec2.get_all_instances()
cmd, instance = aws_tools.get_ec2_instance()


#cmd = boto.manage.cmdshell.sshclient_from_instance(instance, 
#	'/home/gpanterov/.ssh/' + key_name + '.pem', user_name=user_name)
#cmd, instance = aws_tools.get_ec2_instance()
contents = aws_tools.file_to_string(bucket_name='georgipanterov.ga.data.patterns.test.bucket',
									key_name='test_gae2')
#bucket_name = 'georgipanterov.data.patterns'
#key_name = 'data_file.csv'
#aws_tools.create_bucket(bucket_name)
#key = aws_tools.store_private_data(bucket_name, key_name, 
#	'sample_data1.csv')

	
#key2 = aws_tools.download_file(bucket_name, key_name, 's3_data_file.csv')

# To access an ubuntu ami: 
# ssh -i ~/.ssh/aws_ubuntu_key.pem ubuntu@ec2-54-88-216-232.compute-1.amazonaws.com
