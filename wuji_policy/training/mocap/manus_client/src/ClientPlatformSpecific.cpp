// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "ClientPlatformSpecific.hpp"

// signal types
#include <csignal>
// std::filesystem
#include <filesystem>
// std::*fstream
#include <fstream>
// CoreSdk_ShutDown
#include "ManusSDK.h"
// std::map
#include <map>
#include <iostream>
#include <cstring>

/// @brief Reset a signal handler to its default, and then call it.
/// For signal types and explanation, see:
/// https://www.gnu.org/software/libc/manual/html_node/Standard-Signals.html
#define CALL_DEFAULT_SIGNAL_HANDLER(p_SignalType) \
	/* Reset the handler for this signal to the default. */ \
	signal(p_SignalType, SIG_DFL); \
	/* Re-raise this signal, causing the normal handler to run. */ \
	raise(p_SignalType);

const std::string SDKClientPlatformSpecific::s_SlashForFilesystemPath = "/";

/// @brief Handle a signal telling the SDK client to quit.
/// A generic signal used to "politely ask a program to terminate".
/// On Linux, this can be sent by using the Gnome System Monitor and telling
/// the SDK client process to end.
static void HandleTerminationSignal(int p_Parameter)
{
	std::cerr << "Termination signal sent with parameter {}."
		  << p_Parameter << std::endl;

	CoreSdk_ShutDown();

	CALL_DEFAULT_SIGNAL_HANDLER(SIGTERM);
}

/// @brief Handle an interrupt signal.
/// Called when the INTR character is typed - usually ctrl + c.
static void HandleInterruptSignal(int p_Parameter)
{
	std::cerr << "Interrupt signal sent with parameter {}."
		  << p_Parameter << std::endl;

	CoreSdk_ShutDown();

	CALL_DEFAULT_SIGNAL_HANDLER(SIGINT);
}

/// @brief Handle a quit signal.
/// Called when the QUIT character is typed - usually ctrl + \.
static void HandleQuitSignal(int p_Parameter)
{
	std::cerr<< "Quit signal sent with parameter {}."
		<< p_Parameter << std::endl;

	CoreSdk_ShutDown();

	CALL_DEFAULT_SIGNAL_HANDLER(SIGQUIT);
}

/// @brief Handle a hangup signal.
/// Called to report that the user's terminal has disconnected.
/// This can happen when connecting over the network, for example.
/// It also happens if the terminal window is closed while debugging.
static void HandleHangupSignal(int p_Parameter)
{
	std::cerr<< "Hang-up signal sent with parameter {}."
		<< p_Parameter << std::endl;

	CoreSdk_ShutDown();

	CALL_DEFAULT_SIGNAL_HANDLER(SIGHUP);
}


/// @brief Register our signal handling functions.
static bool SetUpSignalHandlers(void)
{
	{
		const __sighandler_t t_OldTerminationHandler = signal(
			SIGTERM,
			HandleTerminationSignal);
		if (t_OldTerminationHandler == SIG_ERR)
		{
			std::cerr <<  ("Failed to set termination signal handler.") << std::endl;
			return false;
		}
	}

	{
		const __sighandler_t t_OldInterruptHandler = signal(
			SIGINT,
			HandleInterruptSignal);
		if (t_OldInterruptHandler == SIG_ERR)
		{
			std::cerr << ("Failed to set interrupt signal handler.") << std::endl;
			return false;
		}
	}

	{
		const __sighandler_t t_OldQuitHandler = signal(
			SIGQUIT,
			HandleQuitSignal);
		if (t_OldQuitHandler == SIG_ERR)
		{
			std::cerr << ("Failed to set quit signal handler.") << std::endl;
			return false;
		}
	}

	{
		const __sighandler_t t_OldHangupHandler = signal(
			SIGHUP,
			HandleHangupSignal);
		if (t_OldHangupHandler == SIG_ERR)
		{
			std::cerr << ("Failed to set hang-up signal handler.")<< std::endl;
			return false;
		}
	}

	return true;
}

bool SDKClientPlatformSpecific::PlatformSpecificInitialization(void)
{
	const bool t_SignalResult = SetUpSignalHandlers();

	return t_SignalResult;
}

bool SDKClientPlatformSpecific::PlatformSpecificShutdown(void)
{
	return true;
}

/*static*/ bool SDKClientPlatformSpecific::CopyString(
	char* const p_Target,
	const size_t p_MaxLengthThatWillFitInTarget,
	const std::string& p_Source)
{
	if (!p_Target)
	{
		std::cerr << 
			"Tried to copy a string, but the target was null. The string was \"{}\"."
			<< p_Source.c_str() << std::endl;

		return false;
	}

	if (p_MaxLengthThatWillFitInTarget == 0)
	{
		std::cerr << 
			"Tried to copy a string, but the target's size is zero. The string was ."
			<< p_Source.c_str() << std::endl;

		return false;
	}

	if (p_MaxLengthThatWillFitInTarget <= p_Source.length())
	{
		std::cerr << 
			"Tried to copy a string that was longer than characters, which makes it too big for its target buffer. The string was "
			<< p_MaxLengthThatWillFitInTarget
			<< p_Source.c_str() << std::endl;

		return false;
	}

	strcpy(p_Target, p_Source.c_str());

	return true;
}


std::string SDKClientPlatformSpecific::GetDocumentsDirectoryPath_UTF8(void)
{
	const char* const t_Xdg = getenv("XDG_DOCUMENTS_DIR");

	// Backup - the documents folder is usually going to be in $HOME/Documents.
	const char* const t_Home = getenv("HOME");
	if (!t_Xdg && !t_Home)
	{
		return std::string("");
	}

	const std::string t_DocumentsDir =
		(!t_Xdg || strlen(t_Xdg) == 0)
			? std::string(t_Home) + std::string("/Documents")
			: std::string(t_Xdg);

	return t_DocumentsDir;
}

std::ifstream SDKClientPlatformSpecific::GetInputFileStream(
	std::string p_Path_UTF8)
{
	return std::ifstream(p_Path_UTF8, std::ifstream::binary);
}

std::ofstream SDKClientPlatformSpecific::GetOutputFileStream(
	std::string p_Path_UTF8)
{
	return std::ofstream(p_Path_UTF8, std::ofstream::binary);
}

bool SDKClientPlatformSpecific::DoesFolderOrFileExist(std::string p_Path_UTF8)
{
	return std::filesystem::exists(p_Path_UTF8);
}

void SDKClientPlatformSpecific::CreateFolderIfItDoesNotExist(
	std::string p_Path_UTF8)
{
	if (!DoesFolderOrFileExist(p_Path_UTF8))
	{
		std::filesystem::create_directory(p_Path_UTF8);
	}
}
