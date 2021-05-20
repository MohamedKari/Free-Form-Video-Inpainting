# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/service/inpainting_service.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='src/service/inpainting_service.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n$src/service/inpainting_service.proto\"3\n\x1eStartInpaintingSessionResponse\x12\x11\n\tSessionId\x18\x01 \x01(\t\"@\n\x0eInpaintRequest\x12\x11\n\tSessionId\x18\x01 \x01(\t\x12\r\n\x05Image\x18\x02 \x01(\x0c\x12\x0c\n\x04Mask\x18\x03 \x01(\x0c\"C\n\x0fInpaintResponse\x12\x16\n\x0eInpaintedFrame\x18\x01 \x01(\x0c\x12\x18\n\x10InpaintedFrameId\x18\x02 \x01(\x05\"3\n\x18ReportBenchmarksResponse\x12\x17\n\x0f\x42\x65nchmarkReport\x18\x01 \x01(\t\"\x07\n\x05\x45mpty2\xd3\x01\n\tInpainter\x12\x18\n\x04Ping\x12\x06.Empty\x1a\x06.Empty\"\x00\x12\x43\n\x16StartInpaintingSession\x12\x06.Empty\x1a\x1f.StartInpaintingSessionResponse\"\x00\x12.\n\x07Inpaint\x12\x0f.InpaintRequest\x1a\x10.InpaintResponse\"\x00\x12\x37\n\x10ReportBenchmarks\x12\x06.Empty\x1a\x19.ReportBenchmarksResponse\"\x00\x62\x06proto3')
)




_STARTINPAINTINGSESSIONRESPONSE = _descriptor.Descriptor(
  name='StartInpaintingSessionResponse',
  full_name='StartInpaintingSessionResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='SessionId', full_name='StartInpaintingSessionResponse.SessionId', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=40,
  serialized_end=91,
)


_INPAINTREQUEST = _descriptor.Descriptor(
  name='InpaintRequest',
  full_name='InpaintRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='SessionId', full_name='InpaintRequest.SessionId', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Image', full_name='InpaintRequest.Image', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Mask', full_name='InpaintRequest.Mask', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=93,
  serialized_end=157,
)


_INPAINTRESPONSE = _descriptor.Descriptor(
  name='InpaintResponse',
  full_name='InpaintResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='InpaintedFrame', full_name='InpaintResponse.InpaintedFrame', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='InpaintedFrameId', full_name='InpaintResponse.InpaintedFrameId', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=159,
  serialized_end=226,
)


_REPORTBENCHMARKSRESPONSE = _descriptor.Descriptor(
  name='ReportBenchmarksResponse',
  full_name='ReportBenchmarksResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='BenchmarkReport', full_name='ReportBenchmarksResponse.BenchmarkReport', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=228,
  serialized_end=279,
)


_EMPTY = _descriptor.Descriptor(
  name='Empty',
  full_name='Empty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=281,
  serialized_end=288,
)

DESCRIPTOR.message_types_by_name['StartInpaintingSessionResponse'] = _STARTINPAINTINGSESSIONRESPONSE
DESCRIPTOR.message_types_by_name['InpaintRequest'] = _INPAINTREQUEST
DESCRIPTOR.message_types_by_name['InpaintResponse'] = _INPAINTRESPONSE
DESCRIPTOR.message_types_by_name['ReportBenchmarksResponse'] = _REPORTBENCHMARKSRESPONSE
DESCRIPTOR.message_types_by_name['Empty'] = _EMPTY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

StartInpaintingSessionResponse = _reflection.GeneratedProtocolMessageType('StartInpaintingSessionResponse', (_message.Message,), dict(
  DESCRIPTOR = _STARTINPAINTINGSESSIONRESPONSE,
  __module__ = 'src.service.inpainting_service_pb2'
  # @@protoc_insertion_point(class_scope:StartInpaintingSessionResponse)
  ))
_sym_db.RegisterMessage(StartInpaintingSessionResponse)

InpaintRequest = _reflection.GeneratedProtocolMessageType('InpaintRequest', (_message.Message,), dict(
  DESCRIPTOR = _INPAINTREQUEST,
  __module__ = 'src.service.inpainting_service_pb2'
  # @@protoc_insertion_point(class_scope:InpaintRequest)
  ))
_sym_db.RegisterMessage(InpaintRequest)

InpaintResponse = _reflection.GeneratedProtocolMessageType('InpaintResponse', (_message.Message,), dict(
  DESCRIPTOR = _INPAINTRESPONSE,
  __module__ = 'src.service.inpainting_service_pb2'
  # @@protoc_insertion_point(class_scope:InpaintResponse)
  ))
_sym_db.RegisterMessage(InpaintResponse)

ReportBenchmarksResponse = _reflection.GeneratedProtocolMessageType('ReportBenchmarksResponse', (_message.Message,), dict(
  DESCRIPTOR = _REPORTBENCHMARKSRESPONSE,
  __module__ = 'src.service.inpainting_service_pb2'
  # @@protoc_insertion_point(class_scope:ReportBenchmarksResponse)
  ))
_sym_db.RegisterMessage(ReportBenchmarksResponse)

Empty = _reflection.GeneratedProtocolMessageType('Empty', (_message.Message,), dict(
  DESCRIPTOR = _EMPTY,
  __module__ = 'src.service.inpainting_service_pb2'
  # @@protoc_insertion_point(class_scope:Empty)
  ))
_sym_db.RegisterMessage(Empty)



_INPAINTER = _descriptor.ServiceDescriptor(
  name='Inpainter',
  full_name='Inpainter',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=291,
  serialized_end=502,
  methods=[
  _descriptor.MethodDescriptor(
    name='Ping',
    full_name='Inpainter.Ping',
    index=0,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_EMPTY,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='StartInpaintingSession',
    full_name='Inpainter.StartInpaintingSession',
    index=1,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_STARTINPAINTINGSESSIONRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Inpaint',
    full_name='Inpainter.Inpaint',
    index=2,
    containing_service=None,
    input_type=_INPAINTREQUEST,
    output_type=_INPAINTRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='ReportBenchmarks',
    full_name='Inpainter.ReportBenchmarks',
    index=3,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_REPORTBENCHMARKSRESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_INPAINTER)

DESCRIPTOR.services_by_name['Inpainter'] = _INPAINTER

# @@protoc_insertion_point(module_scope)
