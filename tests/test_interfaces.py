"""Tests for abc.ABC interface contracts and implementation conformance."""
from __future__ import annotations

import unittest

from vfi_burn.backend import FakeController, MissingWriter
from vfi_burn.ga100 import GA100FuseController, GA100LiveAdapter
from vfi_burn.interfaces import (
    FuseControllerHAL,
    FuseReader,
    FuseWriter,
    MMIOBus,
    MMIOReadBus,
)
from vfi_burn.mmio import AlignedMMIO, ReadOnlyMMIO


class TestInterfaces(unittest.TestCase):
    def test_abstract_interfaces_cannot_be_instantiated_directly(self):
        for iface in (MMIOReadBus, MMIOBus, FuseControllerHAL, FuseReader, FuseWriter):
            with self.subTest(interface=iface.__name__):
                with self.assertRaises(TypeError):
                    iface()

    def test_incomplete_subclass_cannot_be_instantiated(self):
        class PartialMMIO(MMIOBus):
            def read32(self, offset: int) -> int:
                return 0

        class PartialReader(FuseReader):
            def pci_id(self) -> str:
                return "10de:20c2"

        class PartialWriter(FuseWriter):
            pass

        with self.assertRaises(TypeError):
            PartialMMIO()
        with self.assertRaises(TypeError):
            PartialReader()
        with self.assertRaises(TypeError):
            PartialWriter()

    def test_concrete_implementations_inherit_abcs(self):
        buf = bytearray(16)
        ro = ReadOnlyMMIO(buf)
        rw = AlignedMMIO(buf)
        self.assertIsInstance(ro, MMIOReadBus)
        self.assertNotIsInstance(ro, MMIOBus)
        self.assertIsInstance(rw, MMIOReadBus)
        self.assertIsInstance(rw, MMIOBus)

        self.assertTrue(issubclass(GA100FuseController, FuseControllerHAL))
        self.assertTrue(issubclass(GA100LiveAdapter, FuseReader))
        self.assertTrue(issubclass(GA100LiveAdapter, FuseWriter))
        self.assertIsInstance(FakeController(), FuseReader)
        self.assertIsInstance(FakeController(), FuseWriter)
        self.assertIsInstance(MissingWriter(), FuseWriter)
        self.assertNotIsInstance(MissingWriter(), FuseReader)


if __name__ == "__main__":
    unittest.main()
