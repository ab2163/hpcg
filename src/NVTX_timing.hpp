#include <string>
//#include "/opt/nvidia/nsight-systems/2025.3.1/target-linux-x64/nvtx/include/nvtx3/nvtx3.hpp"
#include "/home/abhalerao/nvhpc_2025_255_Linux_x86_64_cuda_12.9/install_components/Linux_x86_64/25.5/profilers/Nsight_Systems/target-linux-x64/nvtx/include/nvtx3/nvtx3.hpp"

#define MAIN_COL 0xFFFF0000
#define MG_COL 0xFFFF7F00
#define DOT_PROD_COL 0xFFFFFF00
#define PROL_COL 0xFF00FF00
#define REST_COL 0xFF00FFFF
#define SPMV_COL 0xFF0000FF
#define SYMGS_COL 0xFF8B00FF
#define WAXPBY_COL 0xFFFF1493

inline void start_timing(const std::string& message, nvtxRangeId_t &range_id){
  if(range_id != 0){
    nvtxRangeEnd(range_id);
  }
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;

  if(message.find("Main") != std::string::npos){
    eventAttrib.color = MAIN_COL;
  }else if(message.find("MG") != std::string::npos){
    eventAttrib.color = MG_COL;
  }else if(message.find("Dot") != std::string::npos){
    eventAttrib.color = DOT_PROD_COL;
  }else if(message.find("Prol") != std::string::npos){
    eventAttrib.color = PROL_COL;
  }else if(message.find("Rest") != std::string::npos){
    eventAttrib.color = REST_COL;
  }else if(message.find("SPMV") != std::string::npos){
    eventAttrib.color = SPMV_COL;
  }else if(message.find("SYMGS") != std::string::npos){
    eventAttrib.color = SYMGS_COL;
  }else if(message.find("WAXPBY") != std::string::npos){
    eventAttrib.color = WAXPBY_COL;
  }else{
    eventAttrib.color = 0xFF888888;
  }

  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message.c_str();
  range_id = nvtxRangeStartEx(&eventAttrib);
}

inline void end_timing(nvtxRangeId_t &range_id){
  nvtxRangeEnd(range_id);
  range_id = 0;
}
