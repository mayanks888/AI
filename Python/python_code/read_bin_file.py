file = open("/home/mayank_sati/Downloads/ti-processor-sdk-rtos-j721e-evm-07_02_00_06/tidl_j7_01_04_00_08/ti_dl/test/tflrt/tflrt-artifacts/attire_mobilenet_v1_224/96_tidl_net.bin", "rb")

byte = file.read(1)
#
# while byte:
#     print(byte)
#     # print(int(byte))
#     byte = file.read(1)
#
# file.close()

d = file.read()
for val in d:
    print(val)
# print(d.decode('UTF-8'))
# print("d[5] = ", d[5]) # d[5] = 40
# print("d[0] = ", d[9]) # d[0] = 128
#
# file.close()
