import boto
from boto.emr.connection import EmrConnection
conn = EmrConnection()
from boto.emr.step import StreamingStep
from boto.emr.bootstrap_action import BootstrapAction

bootstrap_step = BootstrapAction("download.tst", "s3://com.gpanterov.scripts/emr_setup.sh", None)


#mapper='s3n://elasticmapreduce/samples/wordcount/wordSplitter.py'
mapper='s3n://com.gpanterov.scripts/mapper.py'

#reducer = 'aggregate'
reducer='s3n://com.gpanterov.scripts/reducer.py'

step = StreamingStep(name='My Example',
					mapper=mapper,
					reducer=reducer,
					input='s3n://elasticmapreduce/samples/wordcount/input',
					output='s3n://com.gpanterov.outputdata/output/wordcount_output')
jobid = conn.run_jobflow(name='my jobflow', 
		log_uri='s3://com.gpanterov.outputdata/output/jobflow_logs',
		steps=[step],
		bootstrap_actions=[bootstrap_step])

status = conn.describe_jobflow(jobid)
print status.state

