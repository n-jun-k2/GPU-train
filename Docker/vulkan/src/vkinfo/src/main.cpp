#pragma warning(push)
#pragma warning(disable : 26812)
#pragma warning(disable : 4505)

#if defined(ANDROID) || defined (__ANDROID__)
  #define VK_USE_PLATFORM_ANDROID_KHR
#elif defined(_WIN32)
  #define VK_USE_PLATFORM_WIN32_KHR
#else
  #define VK_USE_PLATFORM_XCB_KHR
#endif

#include <iostream>
#include <vulkan/vulkan.h>

int main() {

  VkApplicationInfo app_info = {};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pNext = VK_NULL_HANDLE;
  app_info.pApplicationName = "VULKAN APP";
  app_info.applicationVersion = 1;
  app_info.pEngineName = "VULKAN ENGINE";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  info.pNext = VK_NULL_HANDLE;
  info.flags = 0;
  info.pApplicationInfo = &app_info;
  info.enabledExtensionCount = 0;
  info.ppEnabledExtensionNames = VK_NULL_HANDLE;
  info.enabledLayerCount = 0;
  info.ppEnabledLayerNames = VK_NULL_HANDLE;

  VkInstance instance = VK_NULL_HANDLE;

  VkResult result = vkCreateInstance(&info, VK_NULL_HANDLE, &instance);
  if (result == VK_ERROR_INCOMPATIBLE_DRIVER) {
    std::cout << "cannot find a compatible Vulkan ICD" << std::endl;
  } else if (result) {
    std::cout << "unknown error" << std::endl;
  }

  vkDestroyInstance(instance, VK_NULL_HANDLE);

  std::cout << "vulkan complite" << std::endl;

  return 0;

}