import boto
import boto.manage.cmdshell


def create_bucket(bucket_name):
	s3 = boto.connect_s3(aws_access_key_id='AKIAICBVHKUYHQSPWUAA',
						aws_secret_access_key='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')

	bucket = s3.lookup(bucket_name)
	if bucket:
		print 'Bucket (%s) exists' % bucket_name
	else:
		bucket = s3.create_bucket(bucket_name)
	return bucket

def delete_bucket(bucket_name):
	s3 = boto.connect_s3(aws_access_key_id='AKIAICBVHKUYHQSPWUAA',
					aws_secret_access_key='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')

	bucket = s3.lookup(bucket_name)
	if bucket:
		all_keys = bucket.get_all_keys()
		for key in all_keys:
			key.delete()
		bucket.delete()
	else:
		print "Bucket doesn't exist - nothing to delete"

def check_bucket(bucket_name, num_of_keys):
	s3 = boto.connect_s3(aws_access_key_id='AKIAICBVHKUYHQSPWUAA',
				aws_secret_access_key='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')

	bucket = s3.lookup(bucket_name)
	if bucket:
		all_keys = bucket.get_all_keys()
		if len(all_keys) == num_of_keys:
			return True
		else:
			return False
	else:
		return False

def store_private_data(bucket_name, key_name, path_to_file):
	s3 = boto.connect_s3(aws_access_key_id='AKIAICBVHKUYHQSPWUAA',
						aws_secret_access_key='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')

	bucket = s3.lookup(bucket_name)
	key = bucket.new_key(key_name)
	key.set_contents_from_filename(path_to_file)

	return key

def download_file(bucket_name, key_name, path_to_file):
	s3 = boto.connect_s3(aws_access_key_id='AKIAICBVHKUYHQSPWUAA',
						aws_secret_access_key='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')

	bucket = s3.lookup(bucket_name)
	key = bucket.lookup(key_name)
	key.get_contents_to_filename(path_to_file)
	return key

def file_to_string(bucket_name, key_name):
	s3 = boto.connect_s3(aws_access_key_id='AKIAICBVHKUYHQSPWUAA',
				aws_secret_access_key='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')
	bucket = s3.lookup(bucket_name)
	key = bucket.lookup(key_name)
	contents = key.get_contents_as_string()
	return contents

def get_ec2_instance(path_to_key='/home/gpanterov/.ssh/'):
	ec2 = boto.connect_ec2(aws_access_key_id='AKIAICBVHKUYHQSPWUAA',
						aws_secret_access_key='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')
	reservations = ec2.get_all_instances()
	for i in reservations:
		instance = i.instances[0]
		if instance.public_dns_name == 'ec2-54-88-216-232.compute-1.amazonaws.com':
			break
	
	ec2_key_name = instance.key_name
	print ec2_key_name
	cmd = boto.manage.cmdshell.sshclient_from_instance(instance, 
		path_to_key + ec2_key_name + '.pem', user_name='ubuntu')
	return cmd, instance

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
	ec2 = boto.connect_ec2(aws_access_key_id='AKIAICBVHKUYHQSPWUAA',
						aws_secret_access_key='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')

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

