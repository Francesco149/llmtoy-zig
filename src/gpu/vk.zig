// Single Vulkan @cImport for the whole gpu/ package.
// All gpu/ files import vk from here so opaque handle types unify across modules.
pub const vk = @cImport(@cInclude("vulkan/vulkan.h"));
