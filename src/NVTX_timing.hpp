#include "/opt/nvidia/nsight-systems/2025.3.1/target-linux-x64/nvtx/include/nvtx3/nvtx3.hpp"

#define MAIN_COL 0xFFFF0000
#define MG_COL 0xFFFF7F00
#define DOT_PROD_COL 0xFFFFFF00
#define PROL_COL 0xFF00FF00
#define REST_COL 0xFF00FFFF
#define SPMV_COL 0xFF0000FF
#define SYMGS_COL 0xFF8B00FF
#define WAXPBY_COL 0xFFFF1493
#define ZEROVEC_COL 0xFF00FA9A

void start_timing(char *message, unsigned int color, nvtxRangeId_t &range_id){
  if(range_id != 0){
    nvtxRangeEnd(range_id);
  }
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION; \
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = color;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message;
  range_id = nvtxRangeStartEx(&eventAttrib);
}

void end_timing(nvtxRangeId_t range_id){
  nvtxRangeEnd(range_id);
  range_id = 0;
}