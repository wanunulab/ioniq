import unittest
import PythIon.DataTypes.DataTypes

class DummyTestCase(unittest.TestCase):
    """
    Dummy test case to ensure unittest
    module discovers test cases for this program. 
    Should produce 2 passing tests.
    """
    def setUp(self):
        self.a=10
    def test_a(self):
        self.assertEqual(self.a,10)
        self.a+=1
        self.assertEqual(self.a,11)
    def test_a2(self):
        self.assertEqual(self.a,10)
        
