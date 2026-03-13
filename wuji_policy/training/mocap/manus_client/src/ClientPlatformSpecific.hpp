// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef _CLIENT_PLATFORM_SPECIFIC_HPP_
#define _CLIENT_PLATFORM_SPECIFIC_HPP_

#include "ClientPlatformSpecificTypes.hpp"

#include "ManusSDKTypes.h"

// size_t
#include <cstddef>
// std::string
#include <string>

// Set up a Doxygen group.
/** @addtogroup SDKClient
 *  @{
 */

class SDKClientPlatformSpecific
{
protected:
	/// @brief Initialise things only needed for this platform.
	bool PlatformSpecificInitialization(void);

	/// @brief Shut down things only needed for this platform.
	bool PlatformSpecificShutdown(void);

	/// @brief Copy the given string into the given target.
	static bool CopyString(
		char* const p_Target,
		const size_t p_MaxLengthThatWillFitInTarget,
		const std::string& p_Source);


	/// @brief Get the path to the user's Documents folder.
	/// The string should be in UTF-8 format.
	std::string GetDocumentsDirectoryPath_UTF8(void);

	/// @brief Get an input stream for the given file.
	/// The file's path should be in UTF-8 format.
	std::ifstream GetInputFileStream(std::string p_Path_UTF8);

	/// @brief Get an output stream for the given file.
	/// The file's path should be in UTF-8 format.
	std::ofstream GetOutputFileStream(std::string p_Path_UTF8);

	/// @brief Check if the given folder or file exists.
	/// The folder path given should be in UTF-8 format.
	bool DoesFolderOrFileExist(std::string p_Path_UTF8);

	/// @brief Create the given folder if it does not exist.
	/// The folder path given should be in UTF-8 format.
	void CreateFolderIfItDoesNotExist(std::string p_Path_UTF8);

	/// @brief The slash character that is used in the filesystem.
	static const std::string s_SlashForFilesystemPath;
};

// Close the Doxygen group.
/** @} */

#endif
