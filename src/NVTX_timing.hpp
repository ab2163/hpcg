#include "/opt/nvidia/nsight-systems/2025.3.1/target-linux-x64/nvtx/include/nvtx3/nvtx3.hpp"

nvtxRangeId_t start_timing(char *message, unsigned int color){
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION; \
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = color;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message;
  return nvtxRangeStartEx(&eventAttrib);
}

void start_timing_ref(char *message, unsigned int color, nvtxRangeId_t &range_id_start){
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION; \
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = color;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message;
  range_id_start = nvtxRangeStartEx(&eventAttrib);
}

void end_timing(nvtxRangeId_t range_id_end){
  nvtxRangeEnd(range_id_end);
}

void start_end_timing_ref(char *message, unsigned int color, nvtxRangeId_t range_id_start, nvtxRangeId_t range_id_end){
  end_timing(range_id_end);
  start_timing_ref(message, color, range_id_start);
}