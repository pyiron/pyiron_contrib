import unittest
import os
import json

from moto import mock_s3
import boto3
from pyiron_contrib.generic.s3io import FileS3IO

MY_BUCKET = "MY_BUCKET"
TEST_BUCKET = "TEST_BUCKET"
aws_credentials = {"aws_access_key_id": 'fake_access_key',
                   "aws_secret_access_key": 'fake_secret_key'}
aws_credentials_w_bucket = {"aws_access_key_id": 'fake_access_key',
                            "aws_secret_access_key": 'fake_secret_key',
                            "bucket": MY_BUCKET
                            }
credentials = {"access_key": 'fake_access_key',
               "secret_key": 'fake_secret_key'}
credentials_w_bucket = {"access_key": 'fake_access_key',
                        "secret_key": 'fake_secret_key',
                        "bucket": MY_BUCKET
                        }


class TestS3IO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        with open(cls.current_dir + '/some_file.txt', 'w') as f:
            f.write('text')
        with open(cls.current_dir + '/config.json', 'w') as f:
            json.dump(credentials_w_bucket, f)
        with open(cls.current_dir + '/config_noBucket.json', 'w') as f:
            json.dump(credentials, f)
        cls.moto = mock_s3()
        cls.moto.start()
        cls.res = boto3.resource('s3', **aws_credentials)
        cls.bucket = cls.res.create_bucket(Bucket=MY_BUCKET)
        cls.bucket.put_object(Key='any', Body=b'any text', Metadata={'File_loc': 'root'})
        cls.bucket.put_object(Key='other', Body=b'any text', Metadata={'File_loc': '/'})
        cls.bucket.put_object(Key='other2', Body=b'some text', Metadata={'File_loc': '/'})
        cls.bucket.put_object(Key='some/path', Body=b'some text', Metadata={'File_loc': '/some'})
        cls.bucket.put_object(Key='some/path_to/any', Body=b'any path', Metadata={'File_loc': '/some/path_to'})
        cls.bucket.put_object(Key='some/path_to/some', Body=b'path', Metadata={'File_loc': '/some/path_to'})
        cls.bucket.put_object(Key='random/location', Body=b'text', Metadata={'File_loc': '/random'})
        cls.i_o_bucket = cls.res.create_bucket(Bucket=TEST_BUCKET)
        cls.i_o_bucket.put_object(Key='other', Body=b'any text', Metadata={'File_loc': '/'})
        cls.i_o_bucket.put_object(Key='to_be_removed', Body=b'some text', Metadata={'File_loc': '/'})
        cls.i_o_bucket.put_object(Key='some/path', Body=b'some text', Metadata={'File_loc': '/some'})
        cls.i_o_bucket.put_object(Key='grp_to_be_removed/some', Body=b'foo')
        cls.s3io = FileS3IO(cls.res, bucket_name=MY_BUCKET)
        cls.s3io_io = FileS3IO(cls.res, bucket_name=TEST_BUCKET)

    @classmethod
    def tearDownClass(cls):
        cls.current_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        os.remove(cls.current_dir + '/some_file.txt')
        os.remove(cls.current_dir + '/config.json')
        os.remove(cls.current_dir + '/config_noBucket.json')
        cls.moto = mock_s3()
        cls.res = boto3.resource('s3', **aws_credentials)
        bucket = cls.res.Bucket(MY_BUCKET)
        bucket.objects.all().delete()
        bucket.delete()
        cls.moto.stop()

    def test___init__(self):
        s3io = FileS3IO(self.res, bucket_name=MY_BUCKET)
        self.assertTrue(s3io._bucket == self.bucket)
        self.assertRaises(ValueError, FileS3IO, self.res)
        self.assertRaises(ValueError, FileS3IO, aws_credentials)
        s3io = FileS3IO(aws_credentials, bucket_name=MY_BUCKET)
        self.assertTrue(s3io._bucket == self.bucket)
        s3io = FileS3IO(aws_credentials, bucket_name=TEST_BUCKET)
        self.assertFalse(s3io._bucket == self.bucket)
        s3io = FileS3IO(aws_credentials_w_bucket)
        self.assertTrue(s3io._bucket == self.bucket)
        s3io = FileS3IO(self.current_dir + "/config.json")
        self.assertTrue(s3io._bucket == self.bucket)
        self.assertRaises(ValueError, FileS3IO, self.current_dir + '/config_noBucket.json')
        s3io = FileS3IO(self.current_dir + '/config_noBucket.json', bucket_name=MY_BUCKET)
        self.assertTrue(s3io._bucket == self.bucket)
        self.assertRaises(ValueError, FileS3IO, credentials)
        s3io = FileS3IO(credentials, bucket_name=MY_BUCKET)
        self.assertTrue(s3io._bucket == self.bucket)
        s3io = FileS3IO(credentials_w_bucket)
        self.assertTrue(s3io._bucket == self.bucket)

    def test_list(self):
        self.assertEqual(self.s3io.list_all(), {'groups': ['random', 'some'], 'nodes': ['any', 'other', 'other2']})
        self.assertEqual(self.s3io.list_groups(), ['random', 'some'])
        self.assertEqual(self.s3io.list_nodes(), ['any', 'other', 'other2'])

    def test_get(self):
        other = self.s3io.get('other')
        self.assertEqual(other["Metadata"], {'file_loc': '/'})
        self.assertEqual(other["Body"].read().decode('utf8'), 'any text')

    def test_open(self):
        some = self.s3io.open('some')
        self.assertEqual(some.list_all(), {'groups': ['path_to'], 'nodes': ['path']})
        self.assertEqual(some.s3_path, '/some/')

    def test_close(self):
        opened = self.s3io.open('some')
        opened.close()
        self.assertEqual(opened.s3_path, '/')

    def test_copy(self):
        copy = self.s3io.copy()
        self.assertIsInstance(copy, FileS3IO)
        self.assertEqual(copy.s3_path, '/')
        opened = self.s3io.open('some')
        copy = opened.copy()
        self.assertIsInstance(copy, FileS3IO)
        self.assertEqual(copy.s3_path, '/some/')

    def test_glob(self):
        self.assertEqual(self.s3io.glob('some/path???/*'), ['some/path_to/any', 'some/path_to/some'])

    def test_download(self):
        self.assertFalse(os.path.exists(self.current_dir + '/any'))
        self.s3io.download(['any'], self.current_dir)
        with open('any') as f:
            self.assertEqual(f.read(), 'any text')
        os.remove(self.current_dir + '/any')

    def test___getitem__(self):
        other_s3_obj = self.s3io['other']
        other = other_s3_obj.get()
        self.assertEqual(other["Metadata"], {'file_loc': '/'})
        self.assertEqual(other["Body"].read().decode('utf8'), 'any text')
        self.assertEqual(other_s3_obj.metadata, {'file_loc': '/'})

        some_grp = self.s3io['some']
        self.assertIsInstance(some_grp, FileS3IO)
        self.assertEqual(some_grp.s3_path, '/some/')

        any_s3_obj = self.s3io['some/path_to/any']
        self.assertEqual(any_s3_obj.metadata, {'file_loc': '/some/path_to'})
        self.assertEqual(any_s3_obj.get()['Body'].read().decode('utf8'), 'any path')

        self.assertTrue(self.s3io[""] is self.s3io)

    def test__enter___exit__(self):
        with self.s3io.open('some') as s3:
            self.assertEqual(s3.get_metadata('path'), {'file_loc': '/some'})

    def test_get_metadata(self):
        self.assertEqual(self.s3io.get_metadata('other'), {'file_loc': '/'})

    def test_is_file(self):
        self.assertTrue(self.s3io.is_file('other'))
        self.assertFalse(self.s3io.is_file('some'))

    def test_is_dir(self):
        self.assertTrue(self.s3io.is_dir('some'))
        self.assertFalse(self.s3io.is_dir('other'))

    def test_upload(self):
        file = self.current_dir + '/some_file.txt'
        self.s3io_io.upload([file])
        self.assertTrue(self.s3io_io.is_file('some_file.txt'))
        self.assertEqual(self.s3io_io.get('some_file.txt')['Body'].read().decode('utf8'), 'text')

    def test_remove_file(self):
        self.assertTrue(self.s3io_io.is_file('to_be_removed'))
        self.s3io_io.remove_file('to_be_removed')
        self.assertFalse(self.s3io_io.is_file('to_be_removed'))

    def test_remove_group(self):
        self.assertTrue(self.s3io_io.is_dir('grp_to_be_removed'))
        self.s3io_io.remove_group('grp_to_be_removed')
        self.assertFalse(self.s3io_io.is_dir('grp_to_be_removed'))


if __name__ == '__main__':
    unittest.main()
