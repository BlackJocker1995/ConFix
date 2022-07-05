from Cptool.config import ToolConfig, toolConfig

modelConfig = ToolConfig()
# SITL Type PX4 and Ardupilot
# {'PX4','Ardupilot'}
# SITL类型 PX4 和 Ardupilot
modelConfig.MODE = 'Ardupilot'

# 是否输出Debug信息
modelConfig.DEBUG = False

# Status 长度
modelConfig.STATUS_LEN = len(toolConfig.STATUS_ORDER) - 1

# Parameter的长度
modelConfig.PARAM_LEN = len(toolConfig.PARAM)

# LSTM的输入长度
modelConfig.INPUT_LEN = 4

# LSTM的输出长度
modelConfig.OUTPUT_LEN = 1

# 每一个input数据的长度
modelConfig.DATA_LEN = modelConfig.STATUS_LEN + len(toolConfig.PARAM)

# 输入的数据长度
modelConfig.INPUT_DATA_LEN = modelConfig.DATA_LEN * modelConfig.INPUT_LEN

# 输出的数据长度
modelConfig.OUTPUT_DATA_LEN = modelConfig.STATUS_LEN * modelConfig.OUTPUT_LEN

# 每一个片段的大小
modelConfig.SEGMENT_LEN = 6

# 是否还原
modelConfig.RETRANS = True

modelConfig.EXAMPLE = 244
