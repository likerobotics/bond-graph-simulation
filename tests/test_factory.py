
# test_factory.py
import unittest
from core.base import ElementFactory, ElementType


class TestElementFactory(unittest.TestCase):
    def test_create_resistor(self):
        # Create a resistor and check its type and name format
        r = ElementFactory.create('R')
        self.assertEqual(r.type, ElementType.RESISTOR)
        self.assertTrue(r.name.startswith('R_'))
        self.assertEqual(r.parameter, f"R{r.id}")

    def test_create_capacitor_with_custom_name(self):
        # Create a capacitor with a custom name and check attributes
        c = ElementFactory.create('C', name='CustomCap')
        self.assertEqual(c.type, ElementType.CAPACITOR)
        self.assertEqual(c.name, 'CustomCap')
        self.assertEqual(c.parameter, f"C{c.id}")
        self.assertEqual(c.state_variable, f"q{c.id}")

    def test_create_inductor_state_variable(self):
        # Create an inductor and verify its state variable and parameter
        i = ElementFactory.create('I')
        self.assertEqual(i.state_variable, f"p{i.id}")
        self.assertEqual(i.parameter, f"I{i.id}")

    def test_create_source_effort(self):
        # Create a source of effort and verify effort and input_variable
        se = ElementFactory.create('SE')
        self.assertEqual(se.effort, f"+SE{se.id}")
        self.assertEqual(se.input_variable, f"SE{se.id}")

    def test_create_source_flow(self):
        # Create a source of flow and verify flow and input_variable
        sf = ElementFactory.create('SF')
        self.assertEqual(sf.flow, f"+SF{sf.id}")
        self.assertEqual(sf.input_variable, f"SF{sf.id}")

    def test_create_transformer(self):
        # Create a transformer and verify parameter
        tf = ElementFactory.create('TF')
        self.assertEqual(tf.parameter, f"n{tf.id}")

    def test_create_gyrator(self):
        # Create a gyrator and verify parameter
        gy = ElementFactory.create('GY')
        self.assertEqual(gy.parameter, f"m{gy.id}")

    def test_invalid_type_raises(self):
        # Ensure that an unknown type raises a ValueError
        with self.assertRaises(ValueError):
            ElementFactory.create('XYZ')


if __name__ == '__main__':
    unittest.main()