// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef _SDK_MINIMAL_CLIENT_HPP_
#define _SDK_MINIMAL_CLIENT_HPP_


// Set up a Doxygen group.
/** @addtogroup SDKMinimalClient
 *  @{
 */


#include "ClientPlatformSpecific.hpp"
#include "ManusSDK.h"
#include <mutex>
#include <vector>

/// @brief Values that can be returned by this application.
enum class ClientReturnCode : int
{
	ClientReturnCode_Success = 0,
	ClientReturnCode_FailedPlatformSpecificInitialization,
	ClientReturnCode_FailedToResizeWindow,
	ClientReturnCode_FailedToInitialize,
	ClientReturnCode_FailedToFindHosts,
	ClientReturnCode_FailedToConnect,
	ClientReturnCode_UnrecognizedStateEncountered,
	ClientReturnCode_FailedToShutDownSDK,
	ClientReturnCode_FailedPlatformSpecificShutdown,
	ClientReturnCode_FailedToRestart,
	ClientReturnCode_FailedWrongTimeToGetData,

	ClientReturnCode_MAX_CLIENT_RETURN_CODE_SIZE
};

/// @brief Used to store the information about the final animated skeletons.
class ClientSkeleton
{
public:
	SkeletonInfo info;
	SkeletonNode* nodes = nullptr;

	~ClientSkeleton()
	{
		if (nodes != nullptr)delete[] nodes;
	}
};


/// @brief Used to store the information about the skeleton data coming from the estimation system in Core.
class ClientRawSkeleton
{
public:
	RawSkeletonInfo info;
	std::vector<SkeletonNode> nodes;
};


/// @brief Used to store all the final animated skeletons received from Core.
class ClientSkeletonCollection
{
public:
	std::vector<ClientSkeleton> skeletons;
};


/// @brief Used to store all the skeleton data coming from the estimation system in Core.
class ClientRawSkeletonCollection
{
public:
	std::vector<ClientRawSkeleton> skeletons;
};

/// @brief Used to store all the tracker data coming from Core.
class TrackerDataCollection
{
public:
	std::vector<TrackerData> trackerData;
};

class SDKMinimalClient : public SDKClientPlatformSpecific
{
public:
	SDKMinimalClient();
	~SDKMinimalClient();
	ClientReturnCode Initialize();
	ClientReturnCode InitializeSDK();
	ClientReturnCode ShutDown();
	ClientReturnCode RegisterAllCallbacks();
	void Run();

	static void OnConnectedCallback(const ManusHost* const p_Host);
	static void OnLandscapeCallback(const Landscape* const p_Landscape);
	static void OnSkeletonStreamCallback(const SkeletonStreamInfo* const p_SkeletonStreamInfo);
	static void OnTrackerStreamCallback(const TrackerStreamInfo* const p_TrackerStreamInfo);
	static void OnRawSkeletonStreamCallback(const SkeletonStreamInfo* const p_RawSkeletonStreamInfo);

protected:

	ClientReturnCode Connect();
	ClientReturnCode UpdateBeforeDisplayingData();
	bool SetupHandNodes(uint32_t p_SklIndex);
	bool SetupHandChains(uint32_t p_SklIndex);
	void LoadTestSkeleton();
	void PrintRawSkeletonData();
	void GetRawSkeletonData(std::vector<float>* position, std::vector<float>* quat);
	NodeSetup CreateNodeSetup(uint32_t p_Id, uint32_t p_ParentId, float p_PosX, float p_PosY, float p_PosZ, std::string p_Name);
	static ManusVec3 CreateManusVec3(float p_X, float p_Y, float p_Z);

	static SDKMinimalClient* s_Instance;
	bool m_Running = true;

	std::mutex m_SkeletonMutex;
	std::mutex m_TrackerMutex;
	std::mutex m_RawSkeletonMutex;

	
	ClientRawSkeletonCollection* m_NextRawSkeleton = nullptr;
	ClientRawSkeletonCollection* m_RawSkeleton = nullptr;
	
	ClientSkeletonCollection* m_NextSkeleton = nullptr;
	ClientSkeletonCollection* m_Skeleton = nullptr;

	uint32_t m_FrameCounter = 0;

	// void PrintTrackerData(const TrackerData& trackerData);
	// void PrintTrackerDataGlobal();
	// void PrintTrackerDataPerUser();

	bool m_TrackerTest = false;
	bool m_TrackerDataDisplayPerUser = false;
	float m_TrackerOffset = 0.0f;

	
	TrackerDataCollection* m_NextTrackerData = nullptr;
	TrackerDataCollection* m_TrackerData = nullptr;

	
	std::mutex m_LandscapeMutex;
	Landscape* m_NewLandscape = nullptr;
	Landscape* m_Landscape = nullptr;
	std::vector<GestureLandscapeData> m_NewGestureLandscapeData;
	std::vector<GestureLandscapeData> m_GestureLandscapeData;

	uint32_t m_FirstLeftGloveID = 0;
	uint32_t m_FirstRightGloveID = 0;

};

// Close the Doxygen group.
/** @} */
#endif