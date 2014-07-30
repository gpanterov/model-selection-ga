import boto
import time
import boto.manage.cmdshell
import os
import aws_tools as aws_tools

reload(aws_tools)
#ec2 = boto.connect_ec2()
#ec2 = boto.connect_ec2(aws_access_key_id='AKIAICBVHKUYHQSPWUAA1',
#	aws_secret_access_key='HLTjJI6HTd+BYI68UeYfD5hTSknkeZbmsgZhTeNh')

def launch_instance(ami='ami-7341831a',
					instance_type='t1.micro',
					key_name='paws',
					key_extension='.pem',
					key_dir='~/.ssh',
					group_name='paws',
					ssh_port=22,
					cidr='0.0.0.0/0',
					tag='paws',
					user_data=None,
					cmd_shell=True,
					login_user='ec2-user',
					ssh_passwd=None):
	cmd = None
	ec2 = boto.connect_ec2()
	try:
		key=ec2.get_all_key_pairs(keynames=[key_name])[0]
	except ec2.ResponseError, e:
		if e.code== 'InvalidKeyPair.NotFound':
			print "Creating keypair: %s" % key_name
			key = ec2.create_key_pair(key_name)
			key.save(key_dir)
		else:
			raise

	try:
		group = ec2.get_all_security_groups(groupnames=[group_name])[0]

	except ec2.ResponseError, e:
		if e.code == 'InvalidGroup.NotFound':	
			print "Creating Security Group: %s" % group_name
			group = ec2.create_security_group(group_name, 
								'A group that allows SSH access')
		else:
			raise

	try:
		group.authorize('tcp', ssh_port, ssh_port, cidr)

	except ec2.ResponseError, e:
		if e.code == 'InvalidPermission.Duplicate':
			print "Security group %s already authorized" % group_name
		else:
			raise

	reservation = ec2.run_instances(ami, 
									key_name=key_name,
									security_groups=[group_name],
									instance_type=instance_type,
									user_data=user_data)

	instance = reservation.instances[0]
	print 'waiting for instance'
	while instance.state !='running':
		print '.'
		time.sleep(5)
		instance.update()
	print 'done'
	instance.add_tag(tag)

	if cmd_shell:
		key_path=os.path.join(os.path.expanduser(key_dir),
							key_name+key_extension)
		cmd = boto.manage.cmdshell.sshclient_from_instance(instance,
														key_path,
														user_name=login_user)
	return (instance, cmd)

AMI = 'ami-6692700e'
key_name='aws_ubuntu_key'
user_name = 'ubuntu'
#instance, cmd = launch_instance(ami=AMI, key_name=key_name)
#ec2 = boto.connect_ec2()
#reservations = ec2.get_all_instances()
#instance = reservations[-1].instances[0]

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
# ssh -i aws_ubuntu_key.pem ubuntu@ec2-54-88-216-232.compute-1.amazonaws.com
