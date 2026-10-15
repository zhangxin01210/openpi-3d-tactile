# GitHub currently serves v2.9.6 with this SHA-512. The pinned registry's
# historical archive hash no longer matches the bytes returned by GitHub.
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO syoyo/tinygltf
    REF "v${VERSION}"
    SHA512 f736b30a55fcbb3b80bf25240e0ab2f50c57c380e1425efc30ce1b987f9167ea4b56d98f89e34947c3d77f1fb0b895c01e2dbfa10ddf7210c320ec55fd77b700
    HEAD_REF master
)

vcpkg_replace_string("${SOURCE_PATH}/tiny_gltf.h" "#include \"json.hpp\"" "#include <nlohmann/json.hpp>")
file(INSTALL "${SOURCE_PATH}/tiny_gltf.h" DESTINATION "${CURRENT_PACKAGES_DIR}/include")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
