//===---- CGOpenMPRuntimeNVPTX.cpp - Interface to OpenMP NVPTX Runtimes ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OpenMP runtime code generation specialized to NVPTX
// targets.
//
//===----------------------------------------------------------------------===//

#include "CGOpenMPRuntimeNVPTX.h"
#include "CGCleanup.h"
#include "CodeGenFunction.h"
#include "ConstantBuilder.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Cuda.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace clang;
using namespace CodeGen;

namespace {
enum OpenMPRTLFunctionNVPTX {
  /// \brief Call to void __kmpc_kernel_init_params(void *ScratchpadPtr);
  OMPRTL_NVPTX__kmpc_kernel_init_params,
  /// \brief Call to void __kmpc_kernel_init(kmp_int32 thread_limit,
  /// int16_t RequiresOMPRuntime);
  OMPRTL_NVPTX__kmpc_kernel_init,
  /// \brief Call to void __kmpc_kernel_deinit(int16_t IsOMPRuntimeInitialized);
  OMPRTL_NVPTX__kmpc_kernel_deinit,
  /// \brief Call to void __kmpc_spmd_kernel_init(kmp_int32 thread_limit,
  /// int16_t RequiresOMPRuntime, int16_t RequiresDataSharing);
  OMPRTL_NVPTX__kmpc_spmd_kernel_init,
  /// \brief Call to void __kmpc_spmd_kernel_deinit();
  OMPRTL_NVPTX__kmpc_spmd_kernel_deinit,
  /// Call to void __kmpc_serialized_parallel(ident_t *loc, kmp_int32
  /// global_tid);
  OMPRTL_NVPTX__kmpc_serialized_parallel,
  /// Call to void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32
  /// global_tid);
  OMPRTL_NVPTX__kmpc_end_serialized_parallel,
  /// \brief Call to void __kmpc_kernel_prepare_parallel(void
  /// *outlined_function, int16_t IsOMPRuntimeInitialized);
  OMPRTL_NVPTX__kmpc_kernel_prepare_parallel,
  /// \brief Call to bool __kmpc_kernel_parallel(void **outlined_function,
  /// int16_t IsOMPRuntimeInitialized);
  OMPRTL_NVPTX__kmpc_kernel_parallel,
  /// \brief Call to void __kmpc_kernel_end_parallel();
  OMPRTL_NVPTX__kmpc_kernel_end_parallel,
  /// \brief Call to bool __kmpc_kernel_convergent_parallel(void *buffer,
  /// uint32_t Mask, bool *IsFinal, kmpc_int32 *LaneSource);
  OMPRTL_NVPTX__kmpc_kernel_convergent_parallel,
  /// \brief Call to void __kmpc_kernel_end_convergent_parallel(void *buffer);
  OMPRTL_NVPTX__kmpc_kernel_end_convergent_parallel,
  /// \brief Call to bool __kmpc_kernel_convergent_simd(
  /// void *buffer, uint32_t Mask, bool *IsFinal, kmpc_int32 *LaneSource,
  /// kmpc_int32 *LaneId, kmpc_int32 *NumLanes);
  OMPRTL_NVPTX__kmpc_kernel_convergent_simd,
  /// \brief Call to void __kmpc_kernel_end_convergent_simd(void *buffer);
  OMPRTL_NVPTX__kmpc_kernel_end_convergent_simd,
  /// \brief Call to void __kmpc_kernel_end_convergent_simd(ident_t *loc,
  /// kmp_int32
  /// global_tid);
  OMPRTL_NVPTX__kmpc_parallel_level,
  /// \brief Call to int32_t __kmpc_warp_active_thread_mask();
  OMPRTL_NVPTX__kmpc_warp_active_thread_mask,
  /// \brief Call to void
  /// __kmpc_initialize_data_sharing_environment(__kmpc_data_sharing_slot
  /// *RootS, size_t InitialDataSize);
  OMPRTL_NVPTX__kmpc_initialize_data_sharing_environment,
  /// \brief Call to void* __kmpc_data_sharing_environment_begin(
  /// __kmpc_data_sharing_slot **SavedSharedSlot, void **SavedSharedStack, void
  /// **SavedSharedFrame, int32_t *SavedActiveThreads, size_t SharingDataSize,
  /// size_t SharingDefaultDataSize, int32_t IsEntryPoint,
  /// int16_t IsOMPRuntimeInitialized);
  OMPRTL_NVPTX__kmpc_data_sharing_environment_begin,
  /// \brief Call to void __kmpc_data_sharing_environment_end(
  /// __kmpc_data_sharing_slot **SavedSharedSlot, void **SavedSharedStack, void
  /// **SavedSharedFrame, int32_t *SavedActiveThreads);
  OMPRTL_NVPTX__kmpc_data_sharing_environment_end,
  /// \brief Call to void* __kmpc_get_data_sharing_environment_frame(int32_t
  /// SourceThreadID, int16_t IsOMPRuntimeInitialized);
  OMPRTL_NVPTX__kmpc_get_data_sharing_environment_frame,
  // Call to void __kmpc_barrier_simple_spmd(ident_t *loc, kmp_int32
  // global_tid);
  OMPRTL_NVPTX__kmpc_barrier_simple_spmd,
  // Call to void __kmpc_barrier_simple_generic(ident_t *loc, kmp_int32
  // global_tid);
  OMPRTL_NVPTX__kmpc_barrier_simple_generic,
  /// \brief Call to __kmpc_nvptx_parallel_reduce_nowait(kmp_int32
  /// global_tid, kmp_int32 num_vars, size_t reduce_size, void* reduce_data,
  /// void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
  /// lane_offset, int16_t shortCircuit),
  /// void (*kmp_InterWarpCopyFctPtr)(void* src, int warp_num));
  OMPRTL_NVPTX__kmpc_parallel_reduce_nowait,
  /// \brief Call to __kmpc_nvptx_parallel_reduce_nowait_simple_spmd(kmp_int32
  /// global_tid, kmp_int32 num_vars, size_t reduce_size, void* reduce_data,
  /// void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
  /// lane_offset, int16_t shortCircuit),
  /// void (*kmp_InterWarpCopyFctPtr)(void* src, int warp_num));
  OMPRTL_NVPTX__kmpc_parallel_reduce_nowait_simple_spmd,
  /// \brief Call to
  /// __kmpc_nvptx_parallel_reduce_nowait_simple_generic(kmp_int32
  /// global_tid, kmp_int32 num_vars, size_t reduce_size, void* reduce_data,
  /// void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
  /// lane_offset, int16_t shortCircuit),
  /// void (*kmp_InterWarpCopyFctPtr)(void* src, int warp_num));
  OMPRTL_NVPTX__kmpc_parallel_reduce_nowait_simple_generic,
  /// \brief Call to __kmpc_nvptx_simd_reduce_nowait(kmp_int32
  /// global_tid, kmp_int32 num_vars, size_t reduce_size, void* reduce_data,
  /// void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
  /// lane_offset, int16_t shortCircuit),
  /// void (*kmp_InterWarpCopyFctPtr)(void* src, int warp_num));
  OMPRTL_NVPTX__kmpc_simd_reduce_nowait,
  /// \brief Call to __kmpc_nvptx_teams_reduce_nowait(int32_t global_tid,
  /// int32_t num_vars, size_t reduce_size, void *reduce_data,
  /// void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
  /// lane_offset, int16_t shortCircuit),
  /// void (*kmp_InterWarpCopyFctPtr)(void* src, int warp_num),
  /// void (*kmp_CopyToScratchpadFctPtr)(void *reduceData, void * scratchpad,
  /// int32_t index, int32_t width),
  /// void (*kmp_LoadReduceFctPtr)(void *reduceData, void * scratchpad, int32_t
  /// index, int32_t width, int32_t reduce))
  OMPRTL_NVPTX__kmpc_teams_reduce_nowait,
  /// \brief Call to __kmpc_nvptx_teams_reduce_nowait_simple_spmd(int32_t
  /// global_tid,
  /// int32_t num_vars, size_t reduce_size, void *reduce_data,
  /// void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
  /// lane_offset, int16_t shortCircuit),
  /// void (*kmp_InterWarpCopyFctPtr)(void* src, int warp_num),
  /// void (*kmp_CopyToScratchpadFctPtr)(void *reduceData, void * scratchpad,
  /// int32_t index, int32_t width),
  /// void (*kmp_LoadReduceFctPtr)(void *reduceData, void * scratchpad, int32_t
  /// index, int32_t width, int32_t reduce))
  OMPRTL_NVPTX__kmpc_teams_reduce_nowait_simple_spmd,
  /// \brief Call to __kmpc_nvptx_teams_reduce_nowait_simple_generic(int32_t
  /// global_tid,
  /// int32_t num_vars, size_t reduce_size, void *reduce_data,
  /// void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
  /// lane_offset, int16_t shortCircuit),
  /// void (*kmp_InterWarpCopyFctPtr)(void* src, int warp_num),
  /// void (*kmp_CopyToScratchpadFctPtr)(void *reduceData, void * scratchpad,
  /// int32_t index, int32_t width),
  /// void (*kmp_LoadReduceFctPtr)(void *reduceData, void * scratchpad, int32_t
  /// index, int32_t width, int32_t reduce))
  OMPRTL_NVPTX__kmpc_teams_reduce_nowait_simple_generic,
  /// \brief Call to __kmpc_nvptx_end_reduce_nowait(int32_t global_tid);
  OMPRTL_NVPTX__kmpc_end_reduce,
  /// \brief Call to __kmpc_nvptx_end_reduce(int32_t global_tid);
  OMPRTL_NVPTX__kmpc_end_reduce_nowait

  //
  //  OMPRTL_NVPTX__kmpc_samuel_print
};

// NVPTX Address space
enum ADDRESS_SPACE {
  ADDRESS_SPACE_SHARED = 3,
};

enum BARRIER {
  CTA_BARRIER = 0,
  PARALLEL_BARRIER = 1,
};

enum STATE_SIZE {
  TASK_STATE_SIZE = 48,
  SIMD_STATE_SIZE = 48,
};

enum DATA_SHARING_SIZES {
  // The maximum number of workers in a kernel.
  DS_Max_Worker_Threads = 992,
  // The size reserved for data in a shared memory slot.
  DS_Slot_Size = 256,
  // The maximum number of threads in a worker warp.
  DS_Max_Worker_Warp_Size = 32,
  // The number of bits required to represent the maximum number of threads in a
  // warp.
  DS_Max_Worker_Warp_Size_Log2 = 5,
  DS_Max_Worker_Warp_Size_Log2_Mask =
      (~0u >> (32 - DS_Max_Worker_Warp_Size_Log2)),
  // The slot size that should be reserved for a working warp.
  DS_Worker_Warp_Slot_Size = DS_Max_Worker_Warp_Size * DS_Slot_Size,
  // the maximum number of teams.
  DS_Max_Teams = 1024,

  // An alternative to the heavy data sharing infrastructure that uses global
  // memory is one that uses device __shared__ memory.  The amount of such space
  // (in bytes) reserved by the OpenMP runtime is noted here.
  DS_SimpleBufferSize = 896,
};

enum COPY_DIRECTION {
  // Global memory to a ReduceData structure
  Global_To_ReduceData,
  // ReduceData structure to Global memory
  ReduceData_To_Global,
  // ReduceData structure to another ReduceData structure
  ReduceData_To_ReduceData,
  // Shuffle instruction result to ReduceData
  Shuffle_To_ReduceData,
};

/// Common pre(post)-action for different OpenMP constructs.
class CommonActionTy final : public PrePostActionTy {
  llvm::Value *EnterCallee;
  ArrayRef<llvm::Value *> EnterArgs;
  llvm::Value *ExitCallee;
  ArrayRef<llvm::Value *> ExitArgs;
  bool Conditional;
  llvm::BasicBlock *ContBlock = nullptr;

public:
  CommonActionTy(llvm::Value *EnterCallee, ArrayRef<llvm::Value *> EnterArgs,
                 llvm::Value *ExitCallee, ArrayRef<llvm::Value *> ExitArgs,
                 bool Conditional = false)
      : EnterCallee(EnterCallee), EnterArgs(EnterArgs), ExitCallee(ExitCallee),
        ExitArgs(ExitArgs), Conditional(Conditional) {}
  void Enter(CodeGenFunction &CGF) override {
    llvm::Value *EnterRes = CGF.EmitRuntimeCall(EnterCallee, EnterArgs);
    if (Conditional) {
      llvm::Value *CallBool = CGF.Builder.CreateIsNotNull(EnterRes);
      auto *ThenBlock = CGF.createBasicBlock("omp_if.then");
      ContBlock = CGF.createBasicBlock("omp_if.end");
      // Generate the branch (If-stmt)
      CGF.Builder.CreateCondBr(CallBool, ThenBlock, ContBlock);
      CGF.EmitBlock(ThenBlock);
    }
  }
  void Done(CodeGenFunction &CGF) {
    // Emit the rest of blocks/branches
    CGF.EmitBranch(ContBlock);
    CGF.EmitBlock(ContBlock, true);
  }
  void Exit(CodeGenFunction &CGF) override {
    CGF.EmitRuntimeCall(ExitCallee, ExitArgs);
  }
};

} // namespace

///
/// NVPTX API calls.
///

/// Get the GPU warp size.
static llvm::Value *GetNVPTXWarpSize(CodeGenFunction &CGF) {
  return CGF.EmitRuntimeCall(
      llvm::Intrinsic::getDeclaration(
          &CGF.CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_warpsize),
      "nvptx_warp_size");
}

/// Get the id of the current thread on the GPU.
static llvm::Value *GetNVPTXThreadID(CodeGenFunction &CGF) {
  return CGF.EmitRuntimeCall(
      llvm::Intrinsic::getDeclaration(
          &CGF.CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x),
      "nvptx_tid");
}

/// Get the id of the warp in the block.
static llvm::Value *GetNVPTXWarpID(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateAShr(GetNVPTXThreadID(CGF), DS_Max_Worker_Warp_Size_Log2,
                        "nvptx_warp_id");
}

/// Get the id of the current thread in the Warp.
static llvm::Value *GetNVPTXThreadWarpID(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateAnd(GetNVPTXThreadID(CGF),
                       Bld.getInt32(DS_Max_Worker_Warp_Size_Log2_Mask));
}

/// Get the id of the current block on the GPU.
static llvm::Value *GetNVPTXBlockID(CodeGenFunction &CGF) {
  return CGF.EmitRuntimeCall(
      llvm::Intrinsic::getDeclaration(
          &CGF.CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x),
      "nvptx_block_id");
}

/// Get the maximum number of threads in a block of the GPU.
static llvm::Value *GetNVPTXNumThreads(CodeGenFunction &CGF) {
  return CGF.EmitRuntimeCall(
      llvm::Intrinsic::getDeclaration(
          &CGF.CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x),
      "nvptx_num_threads");
}

/// Get barrier to synchronize all threads in a block.
static void GetNVPTXCTABarrier(CodeGenFunction &CGF) {
  CGF.EmitRuntimeCall(llvm::Intrinsic::getDeclaration(
      &CGF.CGM.getModule(), llvm::Intrinsic::nvvm_barrier0));
}

/// Get barrier #n to synchronize selected (multiple of 32) threads in
/// a block.
static void GetNVPTXBarrier(CodeGenFunction &CGF, int ID,
                            llvm::Value *NumThreadsVal) {
  CGBuilderTy &Bld = CGF.Builder;
  llvm::Value *Args[] = {Bld.getInt32(ID), NumThreadsVal};
  CGF.EmitRuntimeCall(llvm::Intrinsic::getDeclaration(
                          &CGF.CGM.getModule(), llvm::Intrinsic::nvvm_barrier),
                      Args);
}

/// Synchronize all GPU threads in a block.
static void SyncCTAThreads(CodeGenFunction &CGF) { GetNVPTXCTABarrier(CGF); }

/// Get the value of the thread_limit clause in the teams directive.
/// The runtime always starts thread_limit+warpSize threads.
static llvm::Value *GetThreadLimit(CodeGenFunction &CGF,
                                   bool isSPMDExecutionMode) {
  CGBuilderTy &Bld = CGF.Builder;
  return isSPMDExecutionMode
             ? GetNVPTXNumThreads(CGF)
             : Bld.CreateSub(GetNVPTXNumThreads(CGF), GetNVPTXWarpSize(CGF),
                             "thread_limit");
}

/// Get the thread id of the OMP master thread.
/// The master thread id is the first thread (lane) of the last warp in the
/// GPU block.  Warp size is assumed to be some power of 2.
/// Thread id is 0 indexed.
/// E.g: If NumThreads is 33, master id is 32.
///      If NumThreads is 64, master id is 32.
///      If NumThreads is 1024, master id is 992.
static llvm::Value *GetMasterThreadID(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  llvm::Value *NumThreads = GetNVPTXNumThreads(CGF);

  // We assume that the warp size is a power of 2.
  llvm::Value *Mask = Bld.CreateSub(GetNVPTXWarpSize(CGF), Bld.getInt32(1));

  return Bld.CreateAnd(Bld.CreateSub(NumThreads, Bld.getInt32(1)),
                       Bld.CreateNot(Mask), "master_tid");
}

/// Get number of OMP workers for parallel region after subtracting
/// the master warp.
static llvm::Value *GetNumWorkers(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateNUWSub(GetNVPTXNumThreads(CGF), Bld.getInt32(32),
                          "num_workers");
}

/// Get thread id in team.
/// FIXME: Remove the expensive remainder operation.
static llvm::Value *GetTeamThreadId(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  // N % M = N & (M-1) it M is a power of 2. The master Id is expected to be a
  // power fo two in all cases.
  auto *Mask = Bld.CreateNUWSub(GetMasterThreadID(CGF), Bld.getInt32(1));
  return Bld.CreateAnd(GetNVPTXThreadID(CGF), Mask, "team_tid");
}

/// Get global thread id.
static llvm::Value *GetGlobalThreadId(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateAdd(Bld.CreateMul(GetNVPTXBlockID(CGF), GetNumWorkers(CGF)),
                       GetTeamThreadId(CGF), "global_tid");
}

// \brief Get a 32 bit mask, whose bits set to 1 represent the active threads.
llvm::Value *
CGOpenMPRuntimeNVPTX::getNVPTXWarpActiveThreadsMask(CodeGenFunction &CGF) {
  return CGF.EmitRuntimeCall(
      createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_warp_active_thread_mask),
      None, "warp_active_thread_mask");
}

// \brief Get the number of active threads in a warp.
llvm::Value *
CGOpenMPRuntimeNVPTX::getNVPTXWarpActiveNumThreads(CodeGenFunction &CGF) {
  return CGF.EmitRuntimeCall(
      llvm::Intrinsic::getDeclaration(&CGM.getModule(),
                                      llvm::Intrinsic::nvvm_popc_i),
      getNVPTXWarpActiveThreadsMask(CGF), "warp_active_num_threads");
}

// \brief Get the ID of the thread among the current active threads in the warp.
llvm::Value *
CGOpenMPRuntimeNVPTX::getNVPTXWarpActiveThreadID(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;

  // The active thread Id can be computed as the number of bits in the active
  // mask to the right of the current thread:
  // popc( Mask << (32 - (threadID & 0x1f)) );
  auto *WarpID = GetNVPTXThreadWarpID(CGF);
  auto *Mask = getNVPTXWarpActiveThreadsMask(CGF);
  auto *ShNum = Bld.CreateSub(Bld.getInt32(32), WarpID);
  auto *Sh = Bld.CreateShl(Mask, ShNum);
  return CGF.EmitRuntimeCall(
      llvm::Intrinsic::getDeclaration(&CGM.getModule(),
                                      llvm::Intrinsic::nvvm_popc_i),
      Sh, "warp_active_thread_id");
}

// \brief Get a conditional that is set to true if the thread is the master of
// the active threads in the warp.
llvm::Value *
CGOpenMPRuntimeNVPTX::getNVPTXIsWarpActiveMaster(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateICmpEQ(getNVPTXWarpActiveThreadID(CGF), Bld.getInt32(0),
                          "is_warp_active_master");
}

void CGOpenMPRuntimeNVPTX::emitDistributeStaticInit(
    CodeGenFunction &CGF, SourceLocation Loc,
    OpenMPDistScheduleClauseKind SchedKind, const StaticRTInput &Values,
    bool CoalescedDistSchedule) {
  CGOpenMPRuntime::emitDistributeStaticInit(CGF, Loc, SchedKind, Values,
                                            CoalescedDistSchedule);

  // If we are generating a coalesced schedule for the directive
  // 'target teams distribute parallel for', then the 'distribute' and 'for'
  // parts have been combined and the 'parallel' codegen has been elided.
  // So record here that we are entering a Level 1 parallel region.
  if (CoalescedDistSchedule)
    ParallelNestingLevel++;
}

void CGOpenMPRuntimeNVPTX::emitForStaticFinish(CodeGenFunction &CGF,
                                               SourceLocation Loc,
                                               OpenMPDirectiveKind DKind,
                                               bool CoalescedDistSchedule) {
  // If we are generating a coalesced schedule for the directive
  // 'target teams distribute parallel for', then the 'distribute' and 'for'
  // parts have been combined and the 'parallel' codegen has been elided.
  // So record here that we are leaving a Level 1 parallel region.
  if (CoalescedDistSchedule)
    ParallelNestingLevel--;

  CGOpenMPRuntime::emitForStaticFinish(CGF, Loc, DKind);
}

static FieldDecl *addFieldToRecordDecl(ASTContext &C, DeclContext *DC,
                                       QualType FieldTy) {
  auto *Field = FieldDecl::Create(
      C, DC, SourceLocation(), SourceLocation(), /*Id=*/nullptr, FieldTy,
      C.getTrivialTypeSourceInfo(FieldTy, SourceLocation()),
      /*BW=*/nullptr, /*Mutable=*/false, /*InitStyle=*/ICIS_NoInit);
  Field->setAccess(AS_public);
  DC->addDecl(Field);
  return Field;
}

// \brief Type of the data sharing master slot.
QualType CGOpenMPRuntimeNVPTX::getDataSharingMasterSlotQty() {
  //  struct MasterSlot {
  //    Slot *Next;
  //    void *DataEnd;
  //    char Data[DS_Slot_Size]);
  //  };

  const char *Name = "__openmp_nvptx_data_sharing_master_slot_ty";
  if (DataSharingMasterSlotQty.isNull()) {
    ASTContext &C = CGM.getContext();
    auto *RD = C.buildImplicitRecord(Name);
    RD->startDefinition();
    addFieldToRecordDecl(C, RD, C.getPointerType(getDataSharingSlotQty()));
    addFieldToRecordDecl(C, RD, C.VoidPtrTy);
    llvm::APInt NumElems(C.getTypeSize(C.getUIntPtrType()), DS_Slot_Size);
    QualType DataTy = C.getConstantArrayType(
        C.CharTy, NumElems, ArrayType::Normal, /*IndexTypeQuals=*/0);
    addFieldToRecordDecl(C, RD, DataTy);
    RD->completeDefinition();
    DataSharingMasterSlotQty = C.getRecordType(RD);
  }
  return DataSharingMasterSlotQty;
}

// \brief Type of the data sharing worker warp slot.
QualType CGOpenMPRuntimeNVPTX::getDataSharingWorkerWarpSlotQty() {
  //  struct WorkerWarpSlot {
  //    Slot *Next;
  //    void *DataEnd;
  //    char [DS_Worker_Warp_Slot_Size];
  //  };

  const char *Name = "__openmp_nvptx_data_sharing_worker_warp_slot_ty";
  if (DataSharingWorkerWarpSlotQty.isNull()) {
    ASTContext &C = CGM.getContext();
    auto *RD = C.buildImplicitRecord(Name);
    RD->startDefinition();
    addFieldToRecordDecl(C, RD, C.getPointerType(getDataSharingSlotQty()));
    addFieldToRecordDecl(C, RD, C.VoidPtrTy);
    llvm::APInt NumElems(C.getTypeSize(C.getUIntPtrType()),
                         DS_Worker_Warp_Slot_Size);
    QualType DataTy = C.getConstantArrayType(
        C.CharTy, NumElems, ArrayType::Normal, /*IndexTypeQuals=*/0);
    addFieldToRecordDecl(C, RD, DataTy);
    RD->completeDefinition();
    DataSharingWorkerWarpSlotQty = C.getRecordType(RD);
  }
  return DataSharingWorkerWarpSlotQty;
}

// \brief Get the type of the master or worker slot.
QualType CGOpenMPRuntimeNVPTX::getDataSharingSlotQty(bool UseFixedDataSize,
                                                     bool IsMaster) {
  if (UseFixedDataSize) {
    if (IsMaster)
      return getDataSharingMasterSlotQty();
    return getDataSharingWorkerWarpSlotQty();
  }

  //  struct Slot {
  //    Slot *Next;
  //    void *DataEnd;
  //    char Data[];
  //  };

  const char *Name = "__kmpc_data_sharing_slot";
  if (DataSharingSlotQty.isNull()) {
    ASTContext &C = CGM.getContext();
    auto *RD = C.buildImplicitRecord(Name);
    RD->startDefinition();
    addFieldToRecordDecl(C, RD, C.getPointerType(C.getRecordType(RD)));
    addFieldToRecordDecl(C, RD, C.VoidPtrTy);
    QualType DataTy = C.getIncompleteArrayType(C.CharTy, ArrayType::Normal,
                                               /*IndexTypeQuals=*/0);
    addFieldToRecordDecl(C, RD, DataTy);
    RD->completeDefinition();
    DataSharingSlotQty = C.getRecordType(RD);
  }
  return DataSharingSlotQty;
}

llvm::Type *CGOpenMPRuntimeNVPTX::getDataSharingSlotTy(bool UseFixedDataSize,
                                                       bool IsMaster) {
  return CGM.getTypes().ConvertTypeForMem(
      getDataSharingSlotQty(UseFixedDataSize, IsMaster));
}

// \brief Type of the data sharing root slot.
QualType CGOpenMPRuntimeNVPTX::getDataSharingRootSlotQty() {
  // The type of the global with the root slots:
  //  struct Slots {
  //    MasterSlot MS;
  //    WorkerWarpSlot WS[DS_Max_Worker_Threads/DS_Max_Worker_Warp_Size];
  // };
  if (DataSharingRootSlotQty.isNull()) {
    ASTContext &C = CGM.getContext();
    auto *RD = C.buildImplicitRecord("__openmp_nvptx_data_sharing_ty");
    RD->startDefinition();
    addFieldToRecordDecl(C, RD, getDataSharingMasterSlotQty());
    llvm::APInt NumElems(C.getTypeSize(C.getUIntPtrType()),
                         DS_Max_Worker_Threads / DS_Max_Worker_Warp_Size);
    addFieldToRecordDecl(C, RD, C.getConstantArrayType(
                                    getDataSharingWorkerWarpSlotQty(), NumElems,
                                    ArrayType::Normal, /*IndexTypeQuals=*/0));
    RD->completeDefinition();

    llvm::APInt NumTeams(C.getTypeSize(C.getUIntPtrType()), DS_Max_Teams);
    DataSharingRootSlotQty = C.getConstantArrayType(
        C.getRecordType(RD), NumTeams, ArrayType::Normal, /*IndexTypeQuals=*/0);
  }
  return DataSharingRootSlotQty;
}

// \brief Return address of the initial slot that is used to share data.
LValue CGOpenMPRuntimeNVPTX::getDataSharingRootSlotLValue(CodeGenFunction &CGF,
                                                          bool IsMaster) {
  auto &M = CGM.getModule();

  const char *Name = "__openmp_nvptx_shared_data_slots";
  llvm::GlobalVariable *Gbl = M.getGlobalVariable(Name);

  if (!Gbl) {
    auto *Ty = CGF.getTypes().ConvertTypeForMem(getDataSharingRootSlotQty());
    Gbl = new llvm::GlobalVariable(
        M, Ty,
        /*isConstant=*/false, llvm::GlobalVariable::CommonLinkage,
        llvm::Constant::getNullValue(Ty), Name,
        /*InsertBefore=*/nullptr, llvm::GlobalVariable::NotThreadLocal);
  }

  // Return the master slot if the flag is set, otherwise get the right worker
  // slots.
  if (IsMaster) {
    llvm::Value *Idx[] = {llvm::Constant::getNullValue(CGM.Int32Ty),
                          GetNVPTXBlockID(CGF),
                          llvm::Constant::getNullValue(CGM.Int32Ty)};
    llvm::Value *AddrVal = CGF.Builder.CreateInBoundsGEP(Gbl, Idx);
    return CGF.MakeNaturalAlignAddrLValue(AddrVal,
                                          getDataSharingMasterSlotQty());
  }

  auto *WarpID = GetNVPTXWarpID(CGF);
  llvm::Value *Idx[] = {llvm::Constant::getNullValue(CGM.Int32Ty),
                        GetNVPTXBlockID(CGF),
                        /*WS=*/CGF.Builder.getInt32(1), WarpID};
  llvm::Value *AddrVal = CGF.Builder.CreateInBoundsGEP(Gbl, Idx);
  return CGF.MakeNaturalAlignAddrLValue(AddrVal,
                                        getDataSharingWorkerWarpSlotQty());
}

// \brief Initialize the data sharing slots and pointers.
void CGOpenMPRuntimeNVPTX::initializeDataSharing(CodeGenFunction &CGF,
                                                 bool IsMaster) {
  // We initialized the slot and stack pointer in shared memory with their
  // initial values. Also, we initialize the slots with the initial size.

  auto &Bld = CGF.Builder;
  // auto &Ctx = CGF.getContext();

  // If this is not the OpenMP master thread, make sure that only the warp
  // master does the initialization.
  llvm::BasicBlock *EndBB = CGF.createBasicBlock("after_shared_data_init");

  if (!IsMaster) {
    auto *IsWarpMaster = getNVPTXIsWarpActiveMaster(CGF);
    llvm::BasicBlock *InitBB = CGF.createBasicBlock("shared_data_init");
    Bld.CreateCondBr(IsWarpMaster, InitBB, EndBB);
    CGF.EmitBlock(InitBB);
  }

  auto SlotLV = getDataSharingRootSlotLValue(CGF, IsMaster);

  auto *SlotPtrTy = getDataSharingSlotTy()->getPointerTo();
  auto *CastedSlot =
      Bld.CreateBitCast(SlotLV.getAddress(), SlotPtrTy).getPointer();

  llvm::Value *Args[] = {
      CastedSlot,
      llvm::ConstantInt::get(CGM.SizeTy, IsMaster ? DS_Slot_Size
                                                  : DS_Worker_Warp_Slot_Size)};
  CGF.EmitRuntimeCall(
      createNVPTXRuntimeFunction(
          OMPRTL_NVPTX__kmpc_initialize_data_sharing_environment),
      Args);

  CGF.EmitBlock(EndBB);
  return;
}

// \brief Initialize the data sharing slots and pointers and return the
// generated call.
llvm::Function *CGOpenMPRuntimeNVPTX::createKernelInitializerFunction(
    llvm::Function *WorkerFunction, bool RequiresOMPRuntime) {
  auto &Ctx = CGM.getContext();

  // FIXME: Consider to use name based on the worker function name.
  char Name[] = "__omp_kernel_initialization";

  auto RetQTy = Ctx.getCanonicalType(
      Ctx.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/false));
  auto &CGFI = CGM.getTypes().arrangeLLVMFunctionInfo(
      RetQTy, /*instanceMethod=*/false, /*chainCall=*/false, None,
      FunctionType::ExtInfo(), {}, RequiredArgs::All);

  llvm::Function *InitFn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      Name, &CGM.getModule());

  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, InitFn, CGFI);
  InitFn->setLinkage(llvm::GlobalValue::InternalLinkage);

  CodeGenFunction CGF(CGM, /*suppressNewContext=*/true);
  // We don't need debug information in this function as nothing here refers to
  // user code.
  CGF.disableDebugInfo();
  CGF.StartFunction(GlobalDecl(), RetQTy, InitFn, CGFI, {});

  auto &Bld = CGF.Builder;

  llvm::BasicBlock *MasterBB = CGF.createBasicBlock(".master");
  llvm::BasicBlock *SyncBB = CGF.createBasicBlock(".sync.after.master");
  llvm::BasicBlock *WorkerCheckBB = CGF.createBasicBlock(".isworker");
  llvm::BasicBlock *WorkerBB = CGF.createBasicBlock(".worker");
  llvm::BasicBlock *ExitBB = CGF.createBasicBlock(".exit");

  auto *RetTy = CGM.Int32Ty;
  if (RequiresOMPRuntime) {
    auto *One = llvm::ConstantInt::get(RetTy, 1);
    auto *Zero = llvm::ConstantInt::get(RetTy, 0);
    CGF.EmitStoreOfScalar(Zero, CGF.ReturnValue, /*Volatile=*/false, RetQTy);

    auto *IsMaster =
        Bld.CreateICmpEQ(GetNVPTXThreadID(CGF), GetMasterThreadID(CGF));
    Bld.CreateCondBr(IsMaster, MasterBB, SyncBB);

    CGF.EmitBlock(MasterBB);
    // First action in sequential region:
    // Initialize the state of the OpenMP runtime library on the GPU.
    llvm::Value *InitArgs[] = {GetThreadLimit(CGF, isSPMDExecutionMode()),
                               Bld.getInt16(/*RequiresOMPRuntime=*/1)};
    CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_init), InitArgs);
    initializeDataSharing(CGF, /*IsMaster=*/true);
    CGF.EmitStoreOfScalar(One, CGF.ReturnValue, /*Volatile=*/false, RetQTy);
    CGF.EmitBranch(SyncBB);

    CGF.EmitBlock(SyncBB);
    SyncCTAThreads(CGF);
    CGF.EmitBranch(WorkerCheckBB);

    CGF.EmitBlock(WorkerCheckBB);
    auto *IsWorker = Bld.CreateICmpULT(
        GetNVPTXThreadID(CGF), GetThreadLimit(CGF, isSPMDExecutionMode()));
    Bld.CreateCondBr(IsWorker, WorkerBB, ExitBB);

    CGF.EmitBlock(WorkerBB);
    initializeDataSharing(CGF, /*IsMaster=*/false);
    emitCall(CGF, WorkerFunction);
    CGF.EmitBranch(ExitBB);
  } else {
    // Initialize the state of the OpenMP runtime library on the GPU.
    llvm::Value *InitArgs[] = {GetThreadLimit(CGF, isSPMDExecutionMode()),
                               Bld.getInt16(/*RequiresOMPRuntime=*/0)};
    CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_init), InitArgs);

    auto *IsMaster =
        Bld.CreateICmpEQ(GetNVPTXThreadID(CGF), GetMasterThreadID(CGF));
    auto *RetVal = CGF.EmitScalarConversion(IsMaster, Ctx.BoolTy, RetQTy,
                                            SourceLocation());
    CGF.EmitStoreOfScalar(RetVal, CGF.ReturnValue, /*Volatile=*/false, RetQTy);

    auto *IsWorker = Bld.CreateICmpULT(
        GetNVPTXThreadID(CGF), GetThreadLimit(CGF, isSPMDExecutionMode()));
    Bld.CreateCondBr(IsWorker, WorkerBB, ExitBB);

    CGF.EmitBlock(WorkerBB);
    emitCall(CGF, WorkerFunction);
    CGF.EmitBranch(ExitBB);
  }

  CGF.EmitBlock(ExitBB);
  CGF.FinishFunction();

  return InitFn;
}

namespace {
class WithinDeclareTargetRAII {
private:
  bool &IsWithinDeclareTarget;
  bool Prev;

public:
  WithinDeclareTargetRAII(bool &IsWithinDeclareTarget)
      : IsWithinDeclareTarget(IsWithinDeclareTarget),
        Prev(IsWithinDeclareTarget) {
    IsWithinDeclareTarget = true;
  }
  ~WithinDeclareTargetRAII() { IsWithinDeclareTarget = Prev; }
};

//
// Check if the call is to a function in the OpenMP runtime library
// that can be safely called even with the runtime uninitialized.
static bool IsSafeToRemoveRuntime(const FunctionDecl *FD) {
  if (!FD->getDeclName().isIdentifier())
    return false;
  StringRef Name = FD->getName();
  bool RuntimeNotRequired =
      Name.equals("omp_get_wtime") || Name.equals("omp_get_wtick") ||
      Name.equals("omp_get_num_threads") || Name.equals("omp_get_thread_num") ||
      Name.equals("omp_get_num_teams") || Name.equals("omp_get_team_num") ||
      Name.equals("omp_is_initial_device");
  return RuntimeNotRequired;
}

//
// This class is used to traverse a region to see if OMP constructs match the
// given condition.
//
//    If there is an OpenMP construct in the target region, test the condition.
//    There are calls in the target region to externally defined functions and
//    they are not builtins, assume that the condition is met.
//
static void EmptyPrePost(const OMPExecutableDirective &) {}

class OpenMPFinder : public ConstStmtVisitor<OpenMPFinder> {
private:
  using MatchReasonTy = CGOpenMPRuntimeNVPTX::MatchReasonTy;
  using PrePostMatchTy =
      const llvm::function_ref<void(const OMPExecutableDirective &)>;
  using MatchTy = const llvm::function_ref<bool(const OMPExecutableDirective &,
                                                bool, bool &, MatchReasonTy &)>;

  const PrePostMatchTy PreVisit;
  const PrePostMatchTy PostVisit;
  const MatchTy Matcher;
  bool VisitCalls;
  bool MatchesOpenMP;
  bool IsWithinDeclareTarget;
  llvm::SmallPtrSet<const Stmt *, 8> Visited;
  MatchReasonTy MatchReason;

  Stmt *getBody(const FunctionDecl *FD) {
    const FunctionDecl *D = FD;
    if (FunctionTemplateDecl *FunTmpl = FD->getPrimaryTemplate()) {
      if (const FunctionDecl *TD = FunTmpl->getTemplatedDecl())
        D = TD;
    } else if (const FunctionDecl *PD = FD->getTemplateInstantiationPattern()) {
      D = PD;
    }
    return D->doesThisDeclarationHaveABody() ? D->getBody() : nullptr;
  }

  void VisitFunctionDecl(const FunctionDecl *FD, SourceLocation Loc) {
    if (!FD)
      return;
    Stmt *Body = getBody(FD);
    if (Body) {
      // If the call is to a function whose body we have parsed in
      // the frontend, look inside it.
      if (Visited.insert(Body).second) {
        WithinDeclareTargetRAII RAII(IsWithinDeclareTarget);
        Visit(Body);
      }
    } else if (!FD->getBuiltinID()) {
      MatchReason = MatchReasonTy(
          CGOpenMPRuntimeNVPTX::MatchReasonCodeKind::ExternFunctionDefinition,
          Loc);
      // If this is an externally defined function that is not a builtin,
      // assume it matches the requested condition.
      MatchesOpenMP = !IsSafeToRemoveRuntime(FD);
    }
  }
public:
  OpenMPFinder(const MatchTy &Matcher, const PrePostMatchTy &PreVisit,
               const PrePostMatchTy &PostVisit, bool VisitCalls = true)
      : PreVisit(PreVisit), PostVisit(PostVisit), Matcher(Matcher),
        VisitCalls(VisitCalls), MatchesOpenMP(false),
        IsWithinDeclareTarget(false) {}

  void Visit(const Stmt *S) {
    if (MatchesOpenMP)
      return;

    ConstStmtVisitor<OpenMPFinder>::Visit(S);
    for (const Stmt *Child : S->children()) {
      if (Child && !MatchesOpenMP)
        Visit(Child);
    }
  }

  void VisitCallExpr(const CallExpr *E) {
    if (!VisitCalls)
      return;

    VisitFunctionDecl(E->getDirectCallee(), E->getLocStart());
  }

  void VisitCXXConstructExpr(const CXXConstructExpr *E) {
    if (!VisitCalls)
      return;

    VisitFunctionDecl(E->getConstructor(), E->getLocStart());

    // Visit the destructor.
    QualType Ty = E->getType();
    const CXXRecordDecl *RD = Ty->getAsCXXRecordDecl();
    if (!MatchesOpenMP && RD)
      VisitFunctionDecl(RD->getDestructor(), E->getLocStart());
  }

  // Found an OMP directive.
  void VisitOMPExecutableDirective(const Stmt *S) {
    const OMPExecutableDirective &D = *cast<OMPExecutableDirective>(S);
    PreVisit(D);

    bool ShouldVisitDirectiveBody = false;
    MatchesOpenMP = Matcher(D, IsWithinDeclareTarget, ShouldVisitDirectiveBody,
                            MatchReason);

    if (ShouldVisitDirectiveBody && D.hasAssociatedStmt())
      Visit(cast<CapturedStmt>(D.getAssociatedStmt())->getCapturedStmt());

    PostVisit(D);
  }

  bool matchesOpenMP() { return MatchesOpenMP; }
  MatchReasonTy matchReason() { return MatchReason; }
};

/// discard all CompoundStmts intervening between two constructs
static const Stmt *ignoreCompoundStmts(const Stmt *Body) {
  while (auto *CS = dyn_cast_or_null<CompoundStmt>(Body))
    Body = CS->body_front();

  return Body;
}

static bool onlyOneStmt(const Stmt *Body) {
  unsigned size = 1;
  while (auto *CS = dyn_cast_or_null<CompoundStmt>(Body)) {
    Body = CS->body_front();
    size = CS->size();
  }
  return size == 1;
}

// check for inner (nested) SPMD teams construct, if any
static bool hasNestedTeamsSPMDDirective(const OMPExecutableDirective &D,
                                        bool tryCombineAggressively) {
  const auto *PCS = cast<CapturedStmt>(D.getAssociatedStmt());
  if (D.hasClausesOfKind<OMPDependClause>() &&
      isOpenMPTargetExecutionDirective(D.getDirectiveKind()))
    PCS = cast<CapturedStmt>(PCS->getCapturedStmt());
  const CapturedStmt &CS = *PCS;

  if (auto *NestedDir = dyn_cast_or_null<OMPExecutableDirective>(
          ignoreCompoundStmts(CS.getCapturedStmt()))) {
    OpenMPDirectiveKind DirectiveKind = NestedDir->getDirectiveKind();
    if (isOpenMPTeamsDirective(DirectiveKind) &&
        isOpenMPParallelDirective(DirectiveKind)) {
      return true;
    } else if (tryCombineAggressively) {
      if (isOpenMPTeamsDirective(DirectiveKind)) {
        const CapturedStmt &innerCS =
            *cast<CapturedStmt>(NestedDir->getAssociatedStmt());
        if (auto *InnerNestedDir = dyn_cast_or_null<OMPExecutableDirective>(
                ignoreCompoundStmts(innerCS.getCapturedStmt()))) {
          OpenMPDirectiveKind InnerDirectiveKind =
              InnerNestedDir->getDirectiveKind();
          if (onlyOneStmt(CS.getCapturedStmt()) &&
              onlyOneStmt(innerCS.getCapturedStmt()) &&
              isOpenMPDistributeDirective(InnerDirectiveKind) &&
              isOpenMPParallelDirective(InnerDirectiveKind)) {
            return true;
          }
        }
      }
    }
  }

  return false;
}

// check for inner (nested) SPMD teams construct, if any
static const OMPExecutableDirective *
getNestedTeamsSPMDDirective(const OMPExecutableDirective &D,
                            bool tryCombineAggressively) {
  const auto *PCS = cast<CapturedStmt>(D.getAssociatedStmt());
  if (D.hasClausesOfKind<OMPDependClause>() &&
      isOpenMPTargetExecutionDirective(D.getDirectiveKind()))
    PCS = cast<CapturedStmt>(PCS->getCapturedStmt());
  const CapturedStmt &CS = *PCS;

  if (auto *NestedDir = dyn_cast_or_null<OMPExecutableDirective>(
          ignoreCompoundStmts(CS.getCapturedStmt()))) {
    OpenMPDirectiveKind DirectiveKind = NestedDir->getDirectiveKind();
    if (isOpenMPTeamsDirective(DirectiveKind) &&
        isOpenMPParallelDirective(DirectiveKind))
      return NestedDir;
    if (tryCombineAggressively) {
      if (isOpenMPTeamsDirective(DirectiveKind)) {
        const CapturedStmt &innerCS =
            *cast<CapturedStmt>(NestedDir->getAssociatedStmt());
        if (auto *InnerNestedDir = dyn_cast_or_null<OMPExecutableDirective>(
                ignoreCompoundStmts(innerCS.getCapturedStmt()))) {
          OpenMPDirectiveKind InnerDirectiveKind =
              InnerNestedDir->getDirectiveKind();
          if (isOpenMPParallelDirective(InnerDirectiveKind)) {
            return InnerNestedDir;
          }
        }
      }
    }
  }
  return nullptr;
}

static CGOpenMPRuntimeNVPTX::ExecutionMode
GetExecutionMode(const CodeGenModule &CGM, const OMPExecutableDirective &D) {
  if (CGM.getLangOpts().OpenMPNoSPMD)
    return CGOpenMPRuntimeNVPTX::ExecutionMode::GENERIC;

  OpenMPDirectiveKind DirectiveKind = D.getDirectiveKind();
  switch (DirectiveKind) {
  case OMPD_target_simd:
  case OMPD_target: {
    // If the target region as a nested 'teams distribute parallel for',
    // the specifications guarantee that there can be no serial region.
    return hasNestedTeamsSPMDDirective(D,
                                       CGM.getCodeGenOpts().OpenmpCombineDirs)
               ? CGOpenMPRuntimeNVPTX::ExecutionMode::SPMD
               : CGOpenMPRuntimeNVPTX::ExecutionMode::GENERIC;
  }
  case OMPD_target_teams:
  case OMPD_target_teams_distribute:
  case OMPD_target_teams_distribute_simd:
    return CGOpenMPRuntimeNVPTX::ExecutionMode::GENERIC;
  case OMPD_target_parallel:
  case OMPD_target_parallel_for:
  case OMPD_target_parallel_for_simd:
  case OMPD_target_teams_distribute_parallel_for:
  case OMPD_target_teams_distribute_parallel_for_simd:
    return CGOpenMPRuntimeNVPTX::ExecutionMode::SPMD;
  default:
    llvm_unreachable(
        "Unknown programming model for OpenMP directive on NVPTX target.");
  }
  return CGOpenMPRuntimeNVPTX::ExecutionMode::UNKNOWN;
}

static const OMPExecutableDirective *
getSPMDDirective(const CodeGenModule &CGM, const OMPExecutableDirective &D) {
  switch (D.getDirectiveKind()) {
  case OMPD_target:
  case OMPD_target_simd: {
    const OMPExecutableDirective *NestedDir =
        getNestedTeamsSPMDDirective(D, CGM.getCodeGenOpts().OpenmpCombineDirs);
    assert(NestedDir && "Failed to find nested teams SPMD directive.");
    return NestedDir;
  }
  case OMPD_target_parallel:
  case OMPD_target_parallel_for:
  case OMPD_target_parallel_for_simd:
  case OMPD_target_teams_distribute_parallel_for:
  case OMPD_target_teams_distribute_parallel_for_simd:
    return &D;
  default:
    llvm_unreachable("Unknown directive on NVPTX target.");
  }

  return nullptr;
}

static bool worksharingClauseRequiresRuntime(const OMPExecutableDirective &D) {
  // 1. An ordered schedule requires the runtime.
  // 2. Schedule types dynamic, guided, runtime require the runtime.
  OpenMPScheduleClauseKind ScheduleKind = OMPC_SCHEDULE_unknown;
  if (auto *C = D.getSingleClause<OMPScheduleClause>())
    ScheduleKind = C->getScheduleKind();
  return D.getSingleClause<OMPOrderedClause>() != nullptr ||
         ScheduleKind == OMPC_SCHEDULE_dynamic ||
         ScheduleKind == OMPC_SCHEDULE_guided ||
         ScheduleKind == OMPC_SCHEDULE_runtime;
}

static bool directiveRequiresOMPRuntime(const OMPExecutableDirective &D,
                                        bool IsSPMDDirective,
                                        bool IsInParallelRegion) {
  OpenMPDirectiveKind Kind = D.getDirectiveKind();
  if (isOpenMPParallelDirective(Kind)) {
    // num_threads requires the runtime unless it is on a target
    // offload directive.
    // if-clause requires the runtime.
    bool ContainsIfClause = false;
    for (const auto *C : D.getClausesOfKind<OMPIfClause>()) {
      if (C->getNameModifier() == OMPD_parallel ||
          C->getNameModifier() == OMPD_unknown) {
        ContainsIfClause = true;
        break;
      }
    }
    return ((!IsSPMDDirective &&
             (D.getSingleClause<OMPNumThreadsClause>() != nullptr ||
              ContainsIfClause)) ||
            worksharingClauseRequiresRuntime(D));
  } else if (Kind == OMPD_for || Kind == OMPD_for_simd) {
    return !IsInParallelRegion || worksharingClauseRequiresRuntime(D);
  } else if (Kind == OMPD_barrier) {
    return !IsInParallelRegion;
  } else if (Kind == OMPD_teams || Kind == OMPD_distribute ||
             Kind == OMPD_teams_distribute ||
             Kind == OMPD_teams_distribute_simd || Kind == OMPD_atomic) {
    return false;
  } else {
    return true;
  }
}
} // namespace

void CGOpenMPRuntimeNVPTX::TargetKernelProperties::setExecutionMode() {
  Mode = GetExecutionMode(CGM, D);
}

void CGOpenMPRuntimeNVPTX::TargetKernelProperties::setRequiresOMPRuntime() {
  if (CGM.getCodeGenOpts().OpenMPRequireGPURuntime) {
    RequiresOMPRuntime = true;
    return;
  }

  if (Mode == CGOpenMPRuntimeNVPTX::ExecutionMode::SPMD) {
    const OMPExecutableDirective &SD = *getSPMDDirective(CGM, D);
    RequiresOMPRuntime =
        directiveRequiresOMPRuntime(SD, /*IsSPMDDirective=*/true,
                                    /*IsInParallelRegion=*/true);
    RequiresOMPRuntimeReason =
        MatchReasonTy(DirectiveRequiresRuntime, SD.getLocStart());
    if (!RequiresOMPRuntime) {
      auto &&CondGen = [](const OMPExecutableDirective &D, bool, bool &,
                          MatchReasonTy &MatchReason) -> bool {
        OpenMPDirectiveKind Kind = D.getDirectiveKind();
        // Pretty much all directives nested in an SPMD directive require
        // the OpenMP runtime to process.  The only exception is the atomic
        // directive.
        if (Kind == OMPD_atomic)
          return false;
        MatchReason = MatchReasonTy(DirectiveRequiresRuntime, D.getLocStart());
        return true;
      };
      OpenMPFinder Finder(CondGen, [](const OMPExecutableDirective &D) {},
                          [](const OMPExecutableDirective &D) {});
      Finder.Visit(
          cast<CapturedStmt>(SD.getAssociatedStmt())->getCapturedStmt());
      RequiresOMPRuntime = Finder.matchesOpenMP();
      RequiresOMPRuntimeReason = Finder.matchReason();
    }
  } else { // GENERIC
    // Identify nested parallel/simd within lexical scope of the target.
    unsigned ParallelLevel = 0;
    auto &&PreMatch = [&ParallelLevel](const OMPExecutableDirective &D) {
      OpenMPDirectiveKind Kind = D.getDirectiveKind();
      if (isOpenMPParallelDirective(Kind) || Kind == OMPD_simd)
        ParallelLevel++;
    };
    auto &&CondGen = [&ParallelLevel](
        const OMPExecutableDirective &D, bool IsWithinDeclareTarget,
        bool &ShouldVisitDirectiveBody, MatchReasonTy &MatchReason) -> bool {
      // If this directive is inside a declare target function, we
      // need the runtime to handle it.
      if (IsWithinDeclareTarget) {
        MatchReason = MatchReasonTy(DirectiveRequiresRuntime, D.getLocStart());
        return true;
      }

      // We cannot handle nested parallelism without the runtime.
      if (ParallelLevel > 1) {
        MatchReason =
            MatchReasonTy(NestedParallelRequiresRuntime, D.getLocStart());
        return true;
      }

      ShouldVisitDirectiveBody = true;
      if (directiveRequiresOMPRuntime(D, /*IsSPMDDirective=*/false,
                                      ParallelLevel == 1)) {
        MatchReason = MatchReasonTy(DirectiveRequiresRuntime, D.getLocStart());
        return true;
      }
      return false;
    };
    auto &&PostMatch = [&ParallelLevel](const OMPExecutableDirective &D) {
      OpenMPDirectiveKind Kind = D.getDirectiveKind();
      if (isOpenMPParallelDirective(Kind) || Kind == OMPD_simd)
        ParallelLevel--;
    };
    OpenMPFinder Finder(CondGen, PreMatch, PostMatch);
    Finder.Visit(cast<CapturedStmt>(D.getAssociatedStmt())->getCapturedStmt());
    RequiresOMPRuntime = Finder.matchesOpenMP();
    RequiresOMPRuntimeReason = Finder.matchReason();

    if (!RequiresOMPRuntime && MasterSharedDataSize > DS_SimpleBufferSize) {
      RequiresOMPRuntime = true;
      RequiresOMPRuntimeReason =
          MatchReasonTy(MasterContextExceedsSharedMemory, D.getLocStart());
    }
  }
}

// Check if the current target region requires data sharing support.
// Data sharing support is required if this SPMD construct may have a nested
// parallel or simd directive.
void CGOpenMPRuntimeNVPTX::TargetKernelProperties::setRequiresDataSharing() {
  if (CGM.getCodeGenOpts().OpenMPRequireGPURuntime ||
      Mode == CGOpenMPRuntimeNVPTX::ExecutionMode::GENERIC) {
    RequiresDataSharing = true;
    return;
  }

  // Check for a nested 'parallel' (may be combined with other constructs)
  // or 'simd' directive. We only check for the non-combined 'omp simd'
  // directive because we do simd codegen on the gpu for this construct
  // alone. Assume a nested 'parallel' if there is a call to an external
  // function.
  auto &&CondGen = [](const OMPExecutableDirective &D, bool,
                      bool &ShouldVisitDirectiveBody,
                      MatchReasonTy &MatchReason) -> bool {
    OpenMPDirectiveKind Kind = D.getDirectiveKind();
    bool Match = isOpenMPParallelDirective(Kind) || Kind == OMPD_simd;
    ShouldVisitDirectiveBody = !Match;
    return Match;
  };
  OpenMPFinder Finder(CondGen, EmptyPrePost, EmptyPrePost);
  const OMPExecutableDirective &DSPMD = *getSPMDDirective(CGM, D);
  Finder.Visit(
      cast<CapturedStmt>(DSPMD.getAssociatedStmt())->getCapturedStmt());
  RequiresDataSharing = Finder.matchesOpenMP();
}

void CGOpenMPRuntimeNVPTX::TargetKernelProperties::
    setMayContainOrphanedParallel() {
  auto &&CondGen =
      [](const OMPExecutableDirective &D, bool IsWithinDeclareTarget,
         bool &ShouldVisitDirectiveBody, MatchReasonTy &MatchReason) -> bool {
    ShouldVisitDirectiveBody = true;
    return IsWithinDeclareTarget
               ? isOpenMPParallelDirective(D.getDirectiveKind())
               : false;
  };
  OpenMPFinder Finder(CondGen, EmptyPrePost, EmptyPrePost);
  Finder.Visit(cast<CapturedStmt>(D.getAssociatedStmt())->getCapturedStmt());
  MayContainOrphanedParallel = Finder.matchesOpenMP();
}

void CGOpenMPRuntimeNVPTX::TargetKernelProperties::
    setHasAtMostOneNestedParallelInLexicalScope() {
  unsigned NestedParallelCount = 0;
  auto &&CondGen = [&NestedParallelCount](const OMPExecutableDirective &D, bool,
                                          bool &ShouldVisitDirectiveBody,
                                          MatchReasonTy &MatchReason) -> bool {
    ShouldVisitDirectiveBody = true;
    if (isOpenMPParallelDirective(D.getDirectiveKind())) {
      ShouldVisitDirectiveBody = false;
      NestedParallelCount++;
      // If we have found more than one nested parallel region, stop
      // the search.
      if (NestedParallelCount > 1)
        return true;
    }
    return false;
  };
  OpenMPFinder Finder(CondGen, EmptyPrePost, EmptyPrePost,
                      /*VisitCalls=*/false);
  Finder.Visit(cast<CapturedStmt>(D.getAssociatedStmt())->getCapturedStmt());
  HasAtMostOneNestedParallelInLexicalScope = NestedParallelCount <= 1;
}

void CGOpenMPRuntimeNVPTX::TargetKernelProperties::setMasterSharedDataSize() {
  MasterSharedDataSize = 0;
  // Master thread never executes a serial region, so there is no data sharing.
  if (Mode == CGOpenMPRuntimeNVPTX::ExecutionMode::SPMD)
    return;

  if (!D.hasAssociatedStmt())
    return;

  auto &C = CGM.getContext();
  const auto *CS = cast<CapturedStmt>(D.getAssociatedStmt());
  if (D.hasClausesOfKind<OMPDependClause>() &&
      isOpenMPTargetExecutionDirective(D.getDirectiveKind()))
    CS = cast<CapturedStmt>(CS->getCapturedStmt());

  // First get private variables inferred from the target directive's clauses.
  llvm::DenseSet<const VarDecl *> PrivateDecls;
  for (const auto *C : D.getClausesOfKind<OMPPrivateClause>()) {
    for (auto *V : C->varlists()) {
      auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(V)->getDecl());
      PrivateDecls.insert(OrigVD->getCanonicalDecl());
    }
  }
  for (const auto *C : D.getClausesOfKind<OMPFirstprivateClause>()) {
    for (auto *V : C->varlists()) {
      auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(V)->getDecl());
      PrivateDecls.insert(OrigVD->getCanonicalDecl());
    }
  }
  for (const auto *C : D.getClausesOfKind<OMPLastprivateClause>()) {
    for (auto *V : C->varlists()) {
      auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(V)->getDecl());
      PrivateDecls.insert(OrigVD->getCanonicalDecl());
    }
  }
  for (const auto *C : D.getClausesOfKind<OMPReductionClause>()) {
    for (auto *V : C->varlists()) {
      auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(V)->getDecl());
      PrivateDecls.insert(OrigVD->getCanonicalDecl());
    }
  }

  // Mark inputs captured by the target directive.  A nested parallel region
  // may capture references to them.
  llvm::DenseSet<const VarDecl *> InputDecls;
  for (auto &Cap : CS->captures()) {
    if (Cap.capturesVariable() || Cap.capturesVariableByCopy()) {
      auto *V = Cap.getCapturedVar()->getCanonicalDecl();
      if (!PrivateDecls.count(V))
        InputDecls.insert(V);
    }
  }

  SmallVector<const OMPExecutableDirective *, 8> DSDirectives;
  SmallVector<const Stmt *, 64> WorkList;
  WorkList.push_back(CS->getCapturedDecl()->getBody());
  while (!WorkList.empty()) {
    const Stmt *CurStmt = WorkList.pop_back_val();
    if (!CurStmt)
      continue;

    if (auto *Dir = dyn_cast<OMPExecutableDirective>(CurStmt)) {
      if (isOpenMPParallelDirective(Dir->getDirectiveKind()) ||
          isOpenMPSimdDirective(Dir->getDirectiveKind())) {
        DSDirectives.push_back(Dir);
      } else {
        if (Dir->hasAssociatedStmt()) {
          const CapturedStmt &CS =
              *cast<CapturedStmt>(Dir->getAssociatedStmt());
          CurStmt = CS.getCapturedStmt();

          WorkList.push_back(CurStmt);
        }
      }
    } else {
      // Keep looking for other regions.
      WorkList.append(CurStmt->child_begin(), CurStmt->child_end());
    }
  }

  llvm::SmallSet<const VarDecl *, 32> AlreadySharedDecls;
  for (auto *Dir : DSDirectives) {
    const auto *CS = cast<CapturedStmt>(D.getAssociatedStmt());
    if (Dir->hasClausesOfKind<OMPDependClause>() &&
        isOpenMPTargetExecutionDirective(Dir->getDirectiveKind()))
      CS = cast<CapturedStmt>(CS->getCapturedStmt());
    const RecordDecl *RD = CS->getCapturedRecordDecl();
    auto CurField = RD->field_begin();
    auto CurCap = CS->capture_begin();

    for (CapturedStmt::const_capture_init_iterator I = CS->capture_init_begin(),
                                                   E = CS->capture_init_end();
         I != E; ++I, ++CurField, ++CurCap) {
      // Track the data sharing type.
      bool DSRef = false;
      QualType ElemTy;
      if (*I)
        ElemTy = (*I)->getType();
      const VarDecl *CurVD = nullptr;

      if (CurCap->capturesThis()) {
        // We use null to indicate 'this'.
        CurVD = nullptr;
      } else {
        // Get the variable that is initializing the capture.
        if (CurField->hasCapturedVLAType()) {
          ElemTy = C.getSizeType();
          CurVD = ImplicitParamDecl::Create(
              C, /*DC=*/nullptr, CurField->getLocation(), /*Id=*/nullptr,
              ElemTy, ImplicitParamDecl::Other);
        } else {
          CurVD = CurCap->getCapturedVar();
        }

        // If this is an OpenMP capture declaration, we need to look at the
        // original declaration.
        const VarDecl *OrigVD = CurVD;
        if (auto *OD = dyn_cast<OMPCapturedExprDecl>(OrigVD))
          if (auto *DRE =
                  dyn_cast<DeclRefExpr>(OD->getInit()->IgnoreImpCasts()))
            OrigVD = cast<VarDecl>(DRE->getDecl());

        // If the variable does not have local storage it is always a reference.
        // If the variable is a reference, we also share it as is,
        // i.e., consider it a reference to something that can be shared.
        // There are other cases where we may only need to share references, so
        // we may be overestimating data sharing size.
        if (OrigVD->getType()->isReferenceType() || !OrigVD->hasLocalStorage())
          DSRef = true;

        // If the variable is declared outside the target region, only a
        // reference is shared.
        if (InputDecls.count(OrigVD))
          DSRef = true;
      }

      // Do not insert the same declaration twice.
      if (AlreadySharedDecls.count(CurVD))
        continue;
      AlreadySharedDecls.insert(CurVD);

      if (DSRef)
        ElemTy = C.getPointerType(ElemTy);

      unsigned Bytes = C.getTypeSizeInChars(ElemTy).getQuantity();
      MasterSharedDataSize += Bytes;
    }

    // Is this a loop directive?
    if (isOpenMPLoopBoundSharingDirective(Dir->getDirectiveKind())) {
      auto *LD = dyn_cast<OMPLoopDirective>(Dir);
      // Do the bounds of the associated loop need to be shared? This check is
      // the same as checking the existence of an expression that refers to a
      // previous (enclosing) loop.
      if (LD->getPrevLowerBoundVariable()) {
        const VarDecl *LB = cast<VarDecl>(
            cast<DeclRefExpr>(LD->getLowerBoundVariable())->getDecl());
        const VarDecl *UB = cast<VarDecl>(
            cast<DeclRefExpr>(LD->getUpperBoundVariable())->getDecl());

        // Do not insert the same declaration twice.
        if (AlreadySharedDecls.count(LB))
          return;

        // We assume that if the lower bound is not to be shared, the upper
        // bound is not shared as well.
        assert(!AlreadySharedDecls.count(UB) &&
               "Not expecting shared upper bound.");

        assert(LB->getType() == UB->getType() &&
               "Expecting LB and UB to be of same types.");
        QualType ElemTy = LB->getType();
        unsigned Bytes = C.getTypeSizeInChars(ElemTy).getQuantity();
        MasterSharedDataSize += Bytes * 2;

        AlreadySharedDecls.insert(LB);
        AlreadySharedDecls.insert(UB);
      }
    }
  }
}

static const OMPExecutableDirective *
getTeamsDirective(const CodeGenModule &CGM, const OMPExecutableDirective &D) {
  switch (D.getDirectiveKind()) {
  case OMPD_target:
  case OMPD_target_simd: {
    const auto *PCS = cast<CapturedStmt>(D.getAssociatedStmt());
    if (D.hasClausesOfKind<OMPDependClause>() &&
        isOpenMPTargetExecutionDirective(D.getDirectiveKind()))
      PCS = cast<CapturedStmt>(PCS->getCapturedStmt());
    const CapturedStmt &CS = *PCS;
    if (auto *NestedDir = dyn_cast_or_null<OMPExecutableDirective>(
            ignoreCompoundStmts(CS.getCapturedStmt())))
      return NestedDir;
    else
      return &D;
  }
  default:
    return &D;
  }

  return nullptr;
}

void CGOpenMPRuntimeNVPTX::TargetKernelProperties::setTeamsReductionInfo() {
  const OMPExecutableDirective &TD = *getTeamsDirective(CGM, D);

  ReductionVariableCount = 0;
  ReductionSizeInBytes = 0;
  for (const auto *C : TD.getClausesOfKind<OMPReductionClause>()) {
    for (auto *V : C->privates()) {
      assert(!V->getType()->isVariablyModifiedType() &&
             "NVPTX device cannot handle variably modified types.");
      ReductionVariableCount++;
      ReductionSizeInBytes +=
          CGM.getContext().getTypeSize(V->getType().getCanonicalType());
    }
  }
}

void CGOpenMPRuntimeNVPTX::WorkerFunctionState::createWorkerFunction(
    CodeGenModule &CGM) {
  // Create an worker function with no arguments.
  CGFI = &CGM.getTypes().arrangeNullaryFunction();

  WorkerFn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(*CGFI), llvm::GlobalValue::InternalLinkage,
      /* placeholder */ "_worker", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, WorkerFn, *CGFI);
  WorkerFn->setLinkage(llvm::GlobalValue::InternalLinkage);
  WorkerFn->removeFnAttr(llvm::Attribute::NoInline);
  WorkerFn->addFnAttr(llvm::Attribute::AlwaysInline);
}

void CGOpenMPRuntimeNVPTX::emitWorkerFunction(WorkerFunctionState &WST) {
  auto &Ctx = CGM.getContext();

  CodeGenFunction CGF(CGM, /*suppressNewContext=*/true);
  CGF.StartFunction(GlobalDecl(), Ctx.VoidTy, WST.WorkerFn, *WST.CGFI, {},
                    WST.Loc, WST.Loc);
  emitWorkerLoop(CGF, WST);
  CGF.FinishFunction();
}

void CGOpenMPRuntimeNVPTX::emitWorkerLoop(CodeGenFunction &CGF,
                                          WorkerFunctionState &WST) {
  //
  // The workers enter this loop and wait for parallel work from the master.
  // When the master encounters a parallel region it sets up the work + variable
  // arguments, and wakes up the workers.  The workers first check to see if
  // they are required for the parallel region, i.e., within the # of requested
  // parallel threads.  The activated workers load the variable arguments and
  // execute the parallel work.
  //

  CGBuilderTy &Bld = CGF.Builder;

  llvm::BasicBlock *AwaitBB = CGF.createBasicBlock(".await.work");
  llvm::BasicBlock *SelectWorkersBB = CGF.createBasicBlock(".select.workers");
  llvm::BasicBlock *ExecuteBB = CGF.createBasicBlock(".execute.parallel");
  llvm::BasicBlock *TerminateBB = CGF.createBasicBlock(".terminate.parallel");
  llvm::BasicBlock *BarrierBB = CGF.createBasicBlock(".barrier.parallel");
  llvm::BasicBlock *ExitBB = CGF.createBasicBlock(".exit");

  CGF.EmitBranch(AwaitBB);

  // Workers wait for work from master.
  CGF.EmitBlock(AwaitBB);
  // Wait for parallel work
  SyncCTAThreads(CGF);

  Address WorkFn = CGF.CreateTempAlloca(
      CGF.Int8PtrTy, CharUnits::fromQuantity(8), /*Name*/ "work_fn");
  Address ExecStatus =
      CGF.CreateTempAlloca(CGF.Int8Ty, CharUnits::fromQuantity(1),
                           /*Name*/ "exec_status");
  CGF.InitTempAlloca(ExecStatus, Bld.getInt8(/*C=*/0));

  llvm::Value *IsOMPRuntimeInitialized =
      Bld.getInt16(WST.TP.requiresOMPRuntime() ? 1 : 0);
  llvm::Value *Ret = CGF.EmitRuntimeCall(
      createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_parallel),
      {WorkFn.getPointer(), IsOMPRuntimeInitialized});
  Bld.CreateStore(Bld.CreateZExt(Ret, CGF.Int8Ty), ExecStatus);

  llvm::Value *WorkID = Bld.CreateLoad(WorkFn, /*isVolatile=*/true);
  // On termination condition (workfn == 0), exit loop.
  llvm::Value *ShouldTerminate = Bld.CreateICmpEQ(
      WorkID, llvm::Constant::getNullValue(CGF.Int8PtrTy), "should_terminate");
  Bld.CreateCondBr(ShouldTerminate, ExitBB, SelectWorkersBB);

  // Activate requested workers.
  CGF.EmitBlock(SelectWorkersBB);
  llvm::Value *IsActive =
      Bld.CreateICmpNE(Bld.CreateLoad(ExecStatus), Bld.getInt8(0), "is_active");
  Bld.CreateCondBr(IsActive, ExecuteBB, BarrierBB);

  // Signal start of parallel region.
  CGF.EmitBlock(ExecuteBB);

  if (WST.TP.hasAtMostOneL1ParallelRegion()) {
    // Short circuit when there is at most one L1 parallel region.
    assert(!WST.TP.mayContainOrphanedParallel() &&
           "Target region may not contain an orphaned parallel directive.");
    assert(Work.size() <= 1 && "Expecting at most one parallel region.");

    if (Work.size() == 1) {
      // Insert call to work function. We pass the master has source thread ID.
      auto Fn = cast<llvm::Function>(Work[0]);
      emitOutlinedFunctionCall(
          CGF, WST.Loc, Fn,
          {Bld.getInt16(/*ParallelLevel=*/0), GetMasterThreadID(CGF)});
    }

    // Go to end of parallel region.
    CGF.EmitBranch(TerminateBB);
  } else {
    // Process outlined parallel functions in the lexical scope of the target.
    for (auto *W : Work) {
      // Try to match this outlined function.
      auto ThisID = Bld.CreatePtrToInt(W, CGM.Int64Ty);
      ThisID = Bld.CreateIntToPtr(ThisID, CGM.Int8PtrTy);
      llvm::Value *WorkFnMatch = Bld.CreateICmpEQ(WorkID, ThisID, "work_match");

      llvm::BasicBlock *ExecuteFNBB = CGF.createBasicBlock(".execute.fn");
      llvm::BasicBlock *CheckNextBB = CGF.createBasicBlock(".check.next");
      Bld.CreateCondBr(WorkFnMatch, ExecuteFNBB, CheckNextBB);

      // Execute this outlined function.
      CGF.EmitBlock(ExecuteFNBB);

      // Insert call to work function. We pass the master has source thread ID.
      auto Fn = cast<llvm::Function>(W);
      emitOutlinedFunctionCall(
          CGF, WST.Loc, Fn,
          {Bld.getInt16(/*ParallelLevel=*/0), GetMasterThreadID(CGF)});

      // Go to end of parallel region.
      CGF.EmitBranch(TerminateBB);

      CGF.EmitBlock(CheckNextBB);
    }

    // Default case: call to outlined function through pointer if the target
    // region makes a declare target call that may contain an orphaned parallel
    // directive.
    if (WST.TP.mayContainOrphanedParallel()) {
      auto ParallelFnTy =
          llvm::FunctionType::get(CGM.VoidTy, {CGM.Int16Ty, CGM.Int32Ty},
                                  /*isVarArg*/ false)
              ->getPointerTo();
      auto WorkFnCast = Bld.CreateBitCast(WorkID, ParallelFnTy);
      emitOutlinedFunctionCall(
          CGF, WST.Loc, WorkFnCast,
          {Bld.getInt16(/*ParallelLevel=*/0), GetMasterThreadID(CGF)});
      // Go to end of parallel region.
      CGF.EmitBranch(TerminateBB);
    }
  }

  // Signal end of parallel region.
  CGF.EmitBlock(TerminateBB);
  if (WST.TP.requiresOMPRuntime()) {
    CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_end_parallel),
        ArrayRef<llvm::Value *>());
  }
  CGF.EmitBranch(BarrierBB);

  // All active and inactive workers wait at a barrier after parallel region.
  CGF.EmitBlock(BarrierBB);
  // Barrier after parallel region.
  SyncCTAThreads(CGF);
  CGF.EmitBranch(AwaitBB);

  // Exit target region.
  CGF.EmitBlock(ExitBB);
}

// Setup NVPTX threads for master-worker OpenMP scheme.
void CGOpenMPRuntimeNVPTX::emitGenericEntryHeader(CodeGenFunction &CGF,
                                                  EntryFunctionState &EST,
                                                  WorkerFunctionState &WST) {
  //  // Setup BBs in entry function.
  //  llvm::BasicBlock *WorkerCheckBB =
  //  CGF.createBasicBlock(".check.for.worker");
  //  llvm::BasicBlock *WorkerBB = CGF.createBasicBlock(".worker");
  //  llvm::BasicBlock *MasterBB = CGF.createBasicBlock(".master");
  EST.ExitBB = CGF.createBasicBlock(".sleepy.hollow");
  //
  //  // Get the thread limit.
  //  llvm::Value *ThreadLimit = getThreadLimit(CGF);
  //  // Get the master thread id.
  //  llvm::Value *MasterID = getMasterThreadID(CGF);
  //  // Current thread's identifier.
  //  llvm::Value *ThreadID = getNVPTXThreadID(CGF);
  //
  //  // The head (master thread) marches on while its body of companion threads
  //  in
  //  // the warp go to sleep.  Also put to sleep threads in excess of the
  //  // thread_limit value on the teams directive.
  //  llvm::Value *NotMaster = Bld.CreateICmpNE(ThreadID, MasterID,
  //  "not_master");
  //  llvm::Value *ThreadLimitExcess =
  //      Bld.CreateICmpUGE(ThreadID, ThreadLimit, "thread_limit_excess");
  //  llvm::Value *ShouldDie =
  //      Bld.CreateAnd(ThreadLimitExcess, NotMaster, "excess_threads");
  //  Bld.CreateCondBr(ShouldDie, EST.ExitBB, WorkerCheckBB);
  //
  //  // Select worker threads...
  //  CGF.EmitBlock(WorkerCheckBB);
  //  llvm::Value *IsWorker = Bld.CreateICmpULT(ThreadID, MasterID,
  //  "is_worker");
  //  Bld.CreateCondBr(IsWorker, WorkerBB, MasterBB);
  //
  //  // ... and send to worker loop, awaiting parallel invocation.
  //  CGF.EmitBlock(WorkerBB);
  //  llvm::SmallVector<llvm::Value *, 16> WorkerVars;
  //  for (auto &I : CGF.CurFn->args()) {
  //    WorkerVars.push_back(&I);
  //  }
  //
  //  CGF.EmitCallOrInvoke(WST.WorkerFn, None);
  //  CGF.EmitBranch(EST.ExitBB);
  //
  //  // Only master thread executes subsequent serial code.
  //  CGF.EmitBlock(MasterBB);

  // Mark the current function as entry point.
  DataSharingFunctionInfoMap[CGF.CurFn].RequiresOMPRuntime =
      WST.TP.requiresOMPRuntime();
  DataSharingFunctionInfoMap[CGF.CurFn].IsEntryPoint = true;
  DataSharingFunctionInfoMap[CGF.CurFn].EntryWorkerFunction = WST.WorkerFn;
  DataSharingFunctionInfoMap[CGF.CurFn].EntryExitBlock = EST.ExitBB;
}

void CGOpenMPRuntimeNVPTX::emitGenericEntryFooter(CodeGenFunction &CGF,
                                                  EntryFunctionState &EST) {
  if (!EST.ExitBB)
    EST.ExitBB = CGF.createBasicBlock(".exit");

  llvm::BasicBlock *TerminateBB = CGF.createBasicBlock(".termination.notifier");
  CGF.EmitBranch(TerminateBB);

  CGF.EmitBlock(TerminateBB);
  llvm::Value *IsOMPRuntimeInitialized =
      CGF.Builder.getInt16(EST.TP.requiresOMPRuntime() ? 1 : 0);
  // Signal termination condition.
  CGF.EmitRuntimeCall(
      createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_deinit),
      {IsOMPRuntimeInitialized});
  // Barrier to terminate worker threads.
  SyncCTAThreads(CGF);
  // Master thread jumps to exit point.
  CGF.EmitBranch(EST.ExitBB);

  CGF.EmitBlock(EST.ExitBB);
  EST.ExitBB = nullptr;
}

// Create a unique global structure to indicate the execution mode and other
// properties of this target region. This data is picked up by the offload
// library to setup the device before kernel launch.
void CGOpenMPRuntimeNVPTX::SetTargetKernelProperties(
    CodeGenModule &CGM, StringRef TargetName,
    CGOpenMPRuntimeNVPTX::ExecutionMode Mode, unsigned ReductionVariableCount,
    unsigned ReductionSizeInBytes) {
  ASTContext &C = CGM.getContext();

  const char *TTName = "__openmp_nvptx_target_property_ty";
  auto Int32Ty = C.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/true);
  auto *RD = C.buildImplicitRecord(TTName);
  RD->startDefinition();
  // Execution Mode: 0 - SPMD mode, 1 - Generic mode.
  addFieldToRecordDecl(C, RD, C.CharTy);
  // Number of teams reduction variables.
  addFieldToRecordDecl(C, RD, Int32Ty);
  // Total size of reduction variables.
  addFieldToRecordDecl(C, RD, Int32Ty);
  RD->completeDefinition();

  auto *TargetPropertyTy = cast<llvm::StructType>(
      CGM.getTypes().ConvertTypeForMem(C.getRecordType(RD)));

  ConstantInitBuilder Bld(CGM);
  auto fields = Bld.beginStruct(TargetPropertyTy);
  fields.addInt(CGM.Int8Ty,
                Mode == CGOpenMPRuntimeNVPTX::ExecutionMode::GENERIC);
  fields.addInt(CGM.Int32Ty, ReductionVariableCount);
  fields.addInt(CGM.Int32Ty, ReductionSizeInBytes);
  auto Align = CharUnits::fromQuantity(1);
  fields.finishAndCreateGlobal(TargetName + Twine("_property"), Align,
                               /*isConstant=*/true,
                               llvm::GlobalValue::WeakAnyLinkage);
}

void CGOpenMPRuntimeNVPTX::emitSPMDEntryHeader(
    CodeGenFunction &CGF, EntryFunctionState &EST,
    const OMPExecutableDirective &D) {
  auto &Bld = CGF.Builder;
  // Setup BBs in entry function.
  llvm::BasicBlock *ExecuteBB = CGF.createBasicBlock(".execute");
  EST.ExitBB = CGF.createBasicBlock(".sleepy.hollow");

  // Initialize the OMP state in the runtime; called by all active threads.
  llvm::Value *RequiresOMPRuntime =
      Bld.getInt16(EST.TP.requiresOMPRuntime() ? 1 : 0);
  llvm::Value *DS = Bld.getInt16(EST.TP.requiresDataSharing() ? 1 : 0);
  llvm::Value *Args[] = {GetThreadLimit(CGF, isSPMDExecutionMode()),
                         RequiresOMPRuntime, DS};
  CGF.EmitRuntimeCall(
      createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_spmd_kernel_init), Args);
  CGF.EmitBranch(ExecuteBB);

  CGF.EmitBlock(ExecuteBB);
}

void CGOpenMPRuntimeNVPTX::emitSPMDEntryFooter(CodeGenFunction &CGF,
                                               EntryFunctionState &EST) {
  llvm::BasicBlock *OMPDeInitBB = CGF.createBasicBlock(".omp.deinit");
  CGF.EmitBranch(OMPDeInitBB);

  CGF.EmitBlock(OMPDeInitBB);
  if (EST.TP.requiresOMPRuntime()) {
    // DeInitialize the OMP state in the runtime; called by all active threads.
    CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_spmd_kernel_deinit),
        None);
  }
  CGF.EmitBranch(EST.ExitBB);

  CGF.EmitBlock(EST.ExitBB);
}

/// \brief Returns specified OpenMP runtime function for the current OpenMP
/// implementation.  Specialized for the NVPTX device.
/// \param Function OpenMP runtime function.
/// \return Specified function.
llvm::Constant *
CGOpenMPRuntimeNVPTX::createNVPTXRuntimeFunction(unsigned Function) {
  llvm::Constant *RTLFn = nullptr;
  switch (static_cast<OpenMPRTLFunctionNVPTX>(Function)) {
  case OMPRTL_NVPTX__kmpc_kernel_init_params: {
    // Build void __kmpc_kernel_init_params(void *ScratchpadPtr);
    llvm::Type *TypeParams[] = {CGM.Int8PtrTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_init_params");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_init: {
    // Build void __kmpc_kernel_init(kmp_int32 thread_limit,
    // int16_t RequiresOMPRuntime);
    llvm::Type *TypeParams[] = {CGM.Int32Ty, CGM.Int16Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_init");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_deinit: {
    // Build void __kmpc_kernel_deinit(int16_t IsOMPRuntimeInitialized);
    llvm::Type *TypeParams[] = {CGM.Int16Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_deinit");
    break;
  }
  case OMPRTL_NVPTX__kmpc_spmd_kernel_init: {
    // Build void __kmpc_spmd_kernel_init(kmp_int32 thread_limit,
    // int16_t RequiresOMPRuntime, int16_t RequiresDataSharing);
    llvm::Type *TypeParams[] = {CGM.Int32Ty, CGM.Int16Ty, CGM.Int16Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_spmd_kernel_init");
    break;
  }
  case OMPRTL_NVPTX__kmpc_spmd_kernel_deinit: {
    // Build void __kmpc_spmd_kernel_deinit();
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, {}, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_spmd_kernel_deinit");
    break;
  }
  case OMPRTL_NVPTX__kmpc_serialized_parallel: {
    // Build void __kmpc_serialized_parallel(ident_t *loc, kmp_int32
    // global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_serialized_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_end_serialized_parallel: {
    // Build void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32
    // global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_end_serialized_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_prepare_parallel: {
    /// Build void __kmpc_kernel_prepare_parallel(
    /// void *outlined_function, int16_t IsOMPRuntimeInitialized);
    llvm::Type *TypeParams[] = {CGM.Int8PtrTy, CGM.Int16Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_prepare_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_parallel: {
    /// Build bool __kmpc_kernel_parallel(void **outlined_function,
    /// int16_t IsOMPRuntimeInitialized);
    llvm::Type *TypeParams[] = {CGM.Int8PtrPtrTy, CGM.Int16Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(llvm::Type::getInt1Ty(CGM.getLLVMContext()),
                                TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_end_parallel: {
    /// Build void __kmpc_kernel_end_parallel();
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, {}, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_end_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_convergent_parallel: {
    /// \brief Call to bool __kmpc_kernel_convergent_parallel(
    /// void *buffer, uint32_t Mask, bool *IsFinal, kmpc_int32 *LaneSource);
    llvm::Type *TypeParams[] = {CGM.Int8PtrTy, CGM.Int32Ty, CGM.Int8PtrTy,
                                CGM.Int32Ty->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(llvm::Type::getInt1Ty(CGM.getLLVMContext()),
                                TypeParams, /*isVarArg*/ false);
    RTLFn =
        CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_convergent_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_end_convergent_parallel: {
    /// Build void __kmpc_kernel_end_convergent_parallel(void *buffer);
    llvm::Type *TypeParams[] = {CGM.Int8PtrTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy,
                                      "__kmpc_kernel_end_convergent_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_convergent_simd: {
    /// \brief Call to bool __kmpc_kernel_convergent_simd(
    /// void *buffer, uint32_t Mask, bool *IsFinal, kmpc_int32 *LaneSource,
    /// kmpc_int32 *LaneId, kmpc_int32 *NumLanes);
    llvm::Type *TypeParams[] = {CGM.Int8PtrTy,
                                CGM.Int32Ty,
                                CGM.Int8PtrTy,
                                CGM.Int32Ty->getPointerTo(),
                                CGM.Int32Ty->getPointerTo(),
                                CGM.Int32Ty->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(llvm::Type::getInt1Ty(CGM.getLLVMContext()),
                                TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_convergent_simd");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_end_convergent_simd: {
    /// Build void __kmpc_kernel_end_convergent_simd(void *buffer);
    llvm::Type *TypeParams[] = {CGM.Int8PtrTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn =
        CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_end_convergent_simd");
    break;
  }
  case OMPRTL_NVPTX__kmpc_parallel_level: {
    // Build void __kmpc_parallel_level(ident_t *loc, kmp_int32
    // global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int16Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_parallel_level");
    break;
  }
  case OMPRTL_NVPTX__kmpc_warp_active_thread_mask: {
    /// Build void __kmpc_warp_active_thread_mask();
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, None, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_warp_active_thread_mask");
    break;
  }
  case OMPRTL_NVPTX__kmpc_initialize_data_sharing_environment: {
    /// Build void
    /// __kmpc_initialize_data_sharing_environment(__kmpc_data_sharing_slot
    /// *RootS, size_t InitialDataSize);
    auto *SlotTy = CGM.getTypes().ConvertTypeForMem(getDataSharingSlotQty());
    llvm::Type *TypeParams[] = {SlotTy->getPointerTo(), CGM.SizeTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(
        FnTy, "__kmpc_initialize_data_sharing_environment");
    break;
  }
  case OMPRTL_NVPTX__kmpc_data_sharing_environment_begin: {
    /// Build void* __kmpc_data_sharing_environment_begin(
    /// __kmpc_data_sharing_slot **SavedSharedSlot, void **SavedSharedStack,
    /// void **SavedSharedFrame, int32_t *SavedActiveThreads, size_t
    /// SharingDataSize, size_t SharingDefaultDataSize,
    /// int16_t IsOMPRuntimeInitialized);
    auto *SlotTy = CGM.getTypes().ConvertTypeForMem(getDataSharingSlotQty());
    llvm::Type *TypeParams[] = {SlotTy->getPointerTo()->getPointerTo(),
                                CGM.VoidPtrPtrTy,
                                CGM.VoidPtrPtrTy,
                                CGM.Int32Ty->getPointerTo(),
                                CGM.SizeTy,
                                CGM.SizeTy,
                                CGM.Int16Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidPtrTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy,
                                      "__kmpc_data_sharing_environment_begin");
    break;
  }
  case OMPRTL_NVPTX__kmpc_data_sharing_environment_end: {
    /// Build void __kmpc_data_sharing_environment_end( __kmpc_data_sharing_slot
    /// **SavedSharedSlot, void **SavedSharedStack, void **SavedSharedFrame,
    /// int32_t *SavedActiveThreads, int32_t IsEntryPoint);
    auto *SlotTy = CGM.getTypes().ConvertTypeForMem(getDataSharingSlotQty());
    llvm::Type *TypeParams[] = {SlotTy->getPointerTo()->getPointerTo(),
                                CGM.VoidPtrPtrTy, CGM.VoidPtrPtrTy,
                                CGM.Int32Ty->getPointerTo(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn =
        CGM.CreateRuntimeFunction(FnTy, "__kmpc_data_sharing_environment_end");
    break;
  }
  case OMPRTL_NVPTX__kmpc_get_data_sharing_environment_frame: {
    /// Build void* __kmpc_get_data_sharing_environment_frame(int32_t
    /// SourceThreadID, int16_t IsOMPRuntimeInitialized);
    llvm::Type *TypeParams[] = {CGM.Int32Ty, CGM.Int16Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidPtrTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(
        FnTy, "__kmpc_get_data_sharing_environment_frame");
    break;
  }
  case OMPRTL_NVPTX__kmpc_barrier_simple_spmd: {
    // Build void __kmpc_barrier_simple_spmd(ident_t *loc, kmp_int32
    // global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn =
        CGM.CreateRuntimeFunction(FnTy, /*Name*/ "__kmpc_barrier_simple_spmd");
    break;
  }
  case OMPRTL_NVPTX__kmpc_barrier_simple_generic: {
    // Build void __kmpc_barrier_simple_generic(ident_t *loc, kmp_int32
    // global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy,
                                      /*Name*/ "__kmpc_barrier_simple_generic");
    break;
  }
  case OMPRTL_NVPTX__kmpc_parallel_reduce_nowait: {
    /// Build int32_t kmpc_nvptx_parallel_reduce_nowait(kmp_int32 global_tid,
    /// kmp_int32 num_vars, size_t reduce_size, void* reduce_data,
    /// void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
    /// lane_offset, int16_t Algorithm Version),
    /// void (*kmp_InterWarpCopyFctPtr)(void* src, int warp_num));
    llvm::Type *ShuffleReduceTypeParams[] = {CGM.VoidPtrTy, CGM.Int16Ty,
                                             CGM.Int16Ty, CGM.Int16Ty};
    auto *ShuffleReduceFnTy =
        llvm::FunctionType::get(CGM.VoidTy, ShuffleReduceTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *InterWarpCopyTypeParams[] = {CGM.VoidPtrTy, CGM.Int32Ty};
    auto *InterWarpCopyFnTy =
        llvm::FunctionType::get(CGM.VoidTy, InterWarpCopyTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *TypeParams[] = {CGM.Int32Ty,
                                CGM.Int32Ty,
                                CGM.SizeTy,
                                CGM.VoidPtrTy,
                                ShuffleReduceFnTy->getPointerTo(),
                                InterWarpCopyFnTy->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(
        FnTy, /*Name=*/"__kmpc_nvptx_parallel_reduce_nowait");
    break;
  }
  case OMPRTL_NVPTX__kmpc_parallel_reduce_nowait_simple_spmd: {
    /// Build int32_t kmpc_nvptx_parallel_reduce_nowait_simple_spmd(kmp_int32
    /// global_tid,
    /// kmp_int32 num_vars, size_t reduce_size, void* reduce_data,
    /// void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
    /// lane_offset, int16_t Algorithm Version),
    /// void (*kmp_InterWarpCopyFctPtr)(void* src, int warp_num));
    llvm::Type *ShuffleReduceTypeParams[] = {CGM.VoidPtrTy, CGM.Int16Ty,
                                             CGM.Int16Ty, CGM.Int16Ty};
    auto *ShuffleReduceFnTy =
        llvm::FunctionType::get(CGM.VoidTy, ShuffleReduceTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *InterWarpCopyTypeParams[] = {CGM.VoidPtrTy, CGM.Int32Ty};
    auto *InterWarpCopyFnTy =
        llvm::FunctionType::get(CGM.VoidTy, InterWarpCopyTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *TypeParams[] = {CGM.Int32Ty,
                                CGM.Int32Ty,
                                CGM.SizeTy,
                                CGM.VoidPtrTy,
                                ShuffleReduceFnTy->getPointerTo(),
                                InterWarpCopyFnTy->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(
        FnTy, /*Name=*/"__kmpc_nvptx_parallel_reduce_nowait_simple_spmd");
    break;
  }
  case OMPRTL_NVPTX__kmpc_parallel_reduce_nowait_simple_generic: {
    /// Build int32_t kmpc_nvptx_parallel_reduce_nowait_simple_generic(kmp_int32
    /// global_tid,
    /// kmp_int32 num_vars, size_t reduce_size, void* reduce_data,
    /// void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
    /// lane_offset, int16_t Algorithm Version),
    /// void (*kmp_InterWarpCopyFctPtr)(void* src, int warp_num));
    llvm::Type *ShuffleReduceTypeParams[] = {CGM.VoidPtrTy, CGM.Int16Ty,
                                             CGM.Int16Ty, CGM.Int16Ty};
    auto *ShuffleReduceFnTy =
        llvm::FunctionType::get(CGM.VoidTy, ShuffleReduceTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *InterWarpCopyTypeParams[] = {CGM.VoidPtrTy, CGM.Int32Ty};
    auto *InterWarpCopyFnTy =
        llvm::FunctionType::get(CGM.VoidTy, InterWarpCopyTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *TypeParams[] = {CGM.Int32Ty,
                                CGM.Int32Ty,
                                CGM.SizeTy,
                                CGM.VoidPtrTy,
                                ShuffleReduceFnTy->getPointerTo(),
                                InterWarpCopyFnTy->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(
        FnTy, /*Name=*/"__kmpc_nvptx_parallel_reduce_nowait_simple_generic");
    break;
  }
  case OMPRTL_NVPTX__kmpc_simd_reduce_nowait: {
    /// Build int32_t kmpc_nvptx_simd_reduce_nowait(kmp_int32 global_tid,
    /// kmp_int32 num_vars, size_t reduce_size, void* reduce_data,
    /// void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
    /// lane_offset, int16_t Algorithm Version),
    /// void (*kmp_InterWarpCopyFctPtr)(void* src, int warp_num));
    llvm::Type *ShuffleReduceTypeParams[] = {CGM.VoidPtrTy, CGM.Int16Ty,
                                             CGM.Int16Ty, CGM.Int16Ty};
    auto *ShuffleReduceFnTy =
        llvm::FunctionType::get(CGM.VoidTy, ShuffleReduceTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *InterWarpCopyTypeParams[] = {CGM.VoidPtrTy, CGM.Int32Ty};
    auto *InterWarpCopyFnTy =
        llvm::FunctionType::get(CGM.VoidTy, InterWarpCopyTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *TypeParams[] = {CGM.Int32Ty,
                                CGM.Int32Ty,
                                CGM.SizeTy,
                                CGM.VoidPtrTy,
                                ShuffleReduceFnTy->getPointerTo(),
                                InterWarpCopyFnTy->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(
        FnTy, /*Name=*/"__kmpc_nvptx_simd_reduce_nowait");
    break;
  }
  case OMPRTL_NVPTX__kmpc_teams_reduce_nowait: {
    /// Build int32_t __kmpc_nvptx_teams_reduce_nowait(int32_t global_tid,
    /// int32_t num_vars, size_t reduce_size, void *reduce_data,
    /// void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
    /// lane_offset, int16_t shortCircuit),
    /// void (*kmp_InterWarpCopyFctPtr)(void* src, int warp_num),
    /// void (*kmp_CopyToScratchpadFctPtr)(void *reduceData, void * scratchpad,
    /// int32_t index, int32_t width),
    /// void (*kmp_LoadReduceFctPtr)(void *reduceData, void * scratchpad,
    /// int32_t index, int32_t width, int32_t reduce))
    llvm::Type *ShuffleReduceTypeParams[] = {CGM.VoidPtrTy, CGM.Int16Ty,
                                             CGM.Int16Ty, CGM.Int16Ty};
    auto *ShuffleReduceFnTy =
        llvm::FunctionType::get(CGM.VoidTy, ShuffleReduceTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *InterWarpCopyTypeParams[] = {CGM.VoidPtrTy, CGM.Int32Ty};
    auto *InterWarpCopyFnTy =
        llvm::FunctionType::get(CGM.VoidTy, InterWarpCopyTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *CopyToScratchpadTypeParams[] = {CGM.VoidPtrTy, CGM.VoidPtrTy,
                                                CGM.Int32Ty, CGM.Int32Ty};
    auto *CopyToScratchpadFnTy =
        llvm::FunctionType::get(CGM.VoidTy, CopyToScratchpadTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *LoadReduceTypeParams[] = {
        CGM.VoidPtrTy, CGM.VoidPtrTy, CGM.Int32Ty, CGM.Int32Ty, CGM.Int32Ty};
    auto *LoadReduceFnTy =
        llvm::FunctionType::get(CGM.VoidTy, LoadReduceTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *TypeParams[] = {CGM.Int32Ty,
                                CGM.Int32Ty,
                                CGM.SizeTy,
                                CGM.VoidPtrTy,
                                ShuffleReduceFnTy->getPointerTo(),
                                InterWarpCopyFnTy->getPointerTo(),
                                CopyToScratchpadFnTy->getPointerTo(),
                                LoadReduceFnTy->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(
        FnTy, /*Name=*/"__kmpc_nvptx_teams_reduce_nowait");
    break;
  }
  case OMPRTL_NVPTX__kmpc_teams_reduce_nowait_simple_spmd: {
    /// Build int32_t __kmpc_nvptx_teams_reduce_nowait_simple_spmd(int32_t
    /// global_tid,
    /// int32_t num_vars, size_t reduce_size, void *reduce_data,
    /// void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
    /// lane_offset, int16_t shortCircuit),
    /// void (*kmp_InterWarpCopyFctPtr)(void* src, int warp_num),
    /// void (*kmp_CopyToScratchpadFctPtr)(void *reduceData, void * scratchpad,
    /// int32_t index, int32_t width),
    /// void (*kmp_LoadReduceFctPtr)(void *reduceData, void * scratchpad,
    /// int32_t index, int32_t width, int32_t reduce))
    llvm::Type *ShuffleReduceTypeParams[] = {CGM.VoidPtrTy, CGM.Int16Ty,
                                             CGM.Int16Ty, CGM.Int16Ty};
    auto *ShuffleReduceFnTy =
        llvm::FunctionType::get(CGM.VoidTy, ShuffleReduceTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *InterWarpCopyTypeParams[] = {CGM.VoidPtrTy, CGM.Int32Ty};
    auto *InterWarpCopyFnTy =
        llvm::FunctionType::get(CGM.VoidTy, InterWarpCopyTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *CopyToScratchpadTypeParams[] = {CGM.VoidPtrTy, CGM.VoidPtrTy,
                                                CGM.Int32Ty, CGM.Int32Ty};
    auto *CopyToScratchpadFnTy =
        llvm::FunctionType::get(CGM.VoidTy, CopyToScratchpadTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *LoadReduceTypeParams[] = {
        CGM.VoidPtrTy, CGM.VoidPtrTy, CGM.Int32Ty, CGM.Int32Ty, CGM.Int32Ty};
    auto *LoadReduceFnTy =
        llvm::FunctionType::get(CGM.VoidTy, LoadReduceTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *TypeParams[] = {CGM.Int32Ty,
                                CGM.Int32Ty,
                                CGM.SizeTy,
                                CGM.VoidPtrTy,
                                ShuffleReduceFnTy->getPointerTo(),
                                InterWarpCopyFnTy->getPointerTo(),
                                CopyToScratchpadFnTy->getPointerTo(),
                                LoadReduceFnTy->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(
        FnTy, /*Name=*/"__kmpc_nvptx_teams_reduce_nowait_simple_spmd");
    break;
  }
  case OMPRTL_NVPTX__kmpc_teams_reduce_nowait_simple_generic: {
    /// Build int32_t __kmpc_nvptx_teams_reduce_nowait_simple_generic(int32_t
    /// global_tid,
    /// int32_t num_vars, size_t reduce_size, void *reduce_data,
    /// void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
    /// lane_offset, int16_t shortCircuit),
    /// void (*kmp_InterWarpCopyFctPtr)(void* src, int warp_num),
    /// void (*kmp_CopyToScratchpadFctPtr)(void *reduceData, void * scratchpad,
    /// int32_t index, int32_t width),
    /// void (*kmp_LoadReduceFctPtr)(void *reduceData, void * scratchpad,
    /// int32_t index, int32_t width, int32_t reduce))
    llvm::Type *ShuffleReduceTypeParams[] = {CGM.VoidPtrTy, CGM.Int16Ty,
                                             CGM.Int16Ty, CGM.Int16Ty};
    auto *ShuffleReduceFnTy =
        llvm::FunctionType::get(CGM.VoidTy, ShuffleReduceTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *InterWarpCopyTypeParams[] = {CGM.VoidPtrTy, CGM.Int32Ty};
    auto *InterWarpCopyFnTy =
        llvm::FunctionType::get(CGM.VoidTy, InterWarpCopyTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *CopyToScratchpadTypeParams[] = {CGM.VoidPtrTy, CGM.VoidPtrTy,
                                                CGM.Int32Ty, CGM.Int32Ty};
    auto *CopyToScratchpadFnTy =
        llvm::FunctionType::get(CGM.VoidTy, CopyToScratchpadTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *LoadReduceTypeParams[] = {
        CGM.VoidPtrTy, CGM.VoidPtrTy, CGM.Int32Ty, CGM.Int32Ty, CGM.Int32Ty};
    auto *LoadReduceFnTy =
        llvm::FunctionType::get(CGM.VoidTy, LoadReduceTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *TypeParams[] = {CGM.Int32Ty,
                                CGM.Int32Ty,
                                CGM.SizeTy,
                                CGM.VoidPtrTy,
                                ShuffleReduceFnTy->getPointerTo(),
                                InterWarpCopyFnTy->getPointerTo(),
                                CopyToScratchpadFnTy->getPointerTo(),
                                LoadReduceFnTy->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(
        FnTy, /*Name=*/"__kmpc_nvptx_teams_reduce_nowait_simple_generic");
    break;
  }
  case OMPRTL_NVPTX__kmpc_end_reduce: {
    // Build void __kmpc_end_reduce(kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_nvptx_end_reduce");
    break;
  }
  case OMPRTL_NVPTX__kmpc_end_reduce_nowait: {
    // Build __kmpc_end_reduce_nowait(kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(
        FnTy, /*Name=*/"__kmpc_nvptx_end_reduce_nowait");
    break;
  }

    //  case OMPRTL_NVPTX__kmpc_samuel_print: {
    //    llvm::Type *TypeParams[] = {CGM.Int64Ty};
    //    llvm::FunctionType *FnTy =
    //        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/
    //        false);
    //    RTLFn = CGM.CreateRuntimeFunction(
    //        FnTy, "__kmpc_samuel_print");
    //    break;
    //  }
  }
  return RTLFn;
}

llvm::Value *CGOpenMPRuntimeNVPTX::getThreadID(CodeGenFunction &CGF,
                                               SourceLocation Loc) {
  assert(CGF.CurFn && "No function in current CodeGenFunction.");
  return GetGlobalThreadId(CGF);
}

llvm::Value *CGOpenMPRuntimeNVPTX::getParallelLevel(CodeGenFunction &CGF,
                                                    SourceLocation Loc) {
  auto *RTLoc = emitUpdateLocation(CGF, Loc);
  auto ThreadID = getThreadID(CGF, Loc);
  return CGF.EmitRuntimeCall(
      createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_parallel_level),
      {RTLoc, ThreadID});
}

/// \brief Registers the context of a parallel region with the runtime
/// codegen implementation.
void CGOpenMPRuntimeNVPTX::registerParallelContext(
    CodeGenFunction &CGF, const OMPExecutableDirective &S) {
  // Do nothing in case of SPMD Execution Mode and when we are handling the
  // 'parallel' in the SPMD construct.
  if (isSPMDExecutionMode() && InL0())
    return;

  CurrentParallelContext = CGF.CurCodeDecl;

  if (isOpenMPParallelDirective(S.getDirectiveKind()) ||
      isOpenMPSimdDirective(S.getDirectiveKind()))
    createDataSharingInfo(CGF);
}

void CGOpenMPRuntimeNVPTX::createOffloadEntry(llvm::Constant *ID,
                                              llvm::Constant *Addr,
                                              uint64_t Size, uint64_t Flags) {
  auto *F = dyn_cast<llvm::Function>(Addr);
  // TODO: Add support for global variables on the device after declare target
  // support.
  if (!F)
    return;
  llvm::Module *M = F->getParent();
  llvm::LLVMContext &Ctx = M->getContext();

  // Get "nvvm.annotations" metadata node
  llvm::NamedMDNode *MD = M->getOrInsertNamedMetadata("nvvm.annotations");

  llvm::Metadata *MDVals[] = {
      llvm::ConstantAsMetadata::get(F), llvm::MDString::get(Ctx, "kernel"),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), 1))};
  // Append metadata to nvvm.annotations
  MD->addOperand(llvm::MDNode::get(Ctx, MDVals));
}

llvm::Function *
CGOpenMPRuntimeNVPTX::outlineTargetDirective(const OMPExecutableDirective &D,
                                             StringRef Name,
                                             const RegionCodeGenTy &CodeGen) {
  // Generate the outlined function for the target directive.  Wrap it around
  // the offload kernel.  The offload kernel includes NVPTX specific parameters.
  SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  Out << Name << "_impl__";
  llvm::Function *OutlinedFn =
      CGOpenMPRuntime::outlineTargetDirective(D, Out.str(), CodeGen);
  OutlinedFn->removeFnAttr(llvm::Attribute::NoInline);
  OutlinedFn->addFnAttr(llvm::Attribute::AlwaysInline);
  OutlinedFn->setLinkage(llvm::GlobalValue::InternalLinkage);

  //
  // Add device specific parameters to a target region's outlined function.
  // These parameters are added to all target regions.  Arguments for these
  // parameters are passed in by the offload library.
  //
  CodeGenFunction WrapperCGF(CGM, /*suppressNewContext=*/true);
  auto &Ctx = WrapperCGF.getContext();
  const auto *PCS = cast<CapturedStmt>(D.getAssociatedStmt());
  if (D.hasClausesOfKind<OMPDependClause>())
    PCS = cast<CapturedStmt>(PCS->getCapturedStmt());
  const CapturedStmt &CS = *PCS;
  // When a target region has a depend clause, generate a new task
  // that contains the target region invocation, instead of generating it in
  // place. The task will take care of the depend logic.
  bool HasDependClause = D.hasClausesOfKind<OMPDependClause>();
  bool UseCapturedArgumentsOnly =
      isOpenMPParallelDirective(D.getDirectiveKind()) ||
      isOpenMPTeamsDirective(D.getDirectiveKind()) || HasDependClause;
  FunctionArgList Args;
  WrapperCGF.GenerateOpenMPCapturedStmtParameters(
      CS, UseCapturedArgumentsOnly, /*CaptureLevel=*/1, /*ImplicitParamStop=*/0,
      CGM.getCodeGenOpts().OpenmpNonaliasedMaps, /*UIntPtrCastRequired=*/true,
      Args);

  // Pointer to a scratchpad location to store intermediate results of a
  // teams reduction.
  const VarDecl *ScratchpadArg = ImplicitParamDecl::Create(
      Ctx, /*DC=*/nullptr, D.getLocStart(), &Ctx.Idents.get("scratchpad_ptr"),
      Ctx.VoidPtrTy, ImplicitParamDecl::Other);
  Args.emplace_back(ScratchpadArg);

  FunctionType::ExtInfo ExtInfo;
  const CGFunctionInfo &FuncInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(Ctx.VoidTy, Args);
  llvm::FunctionType *FuncLLVMTy = CGM.getTypes().GetFunctionType(FuncInfo);
  llvm::Function *WrapperFn = llvm::Function::Create(
      FuncLLVMTy, llvm::GlobalValue::InternalLinkage, Name, &CGM.getModule());
  const CapturedDecl *CD = CS.getCapturedDecl();
  CGM.SetInternalFunctionAttributes(CD, WrapperFn, FuncInfo);
  if (CD->isNothrow())
    WrapperFn->setDoesNotThrow();

  WrapperCGF.StartFunction(CD, Ctx.VoidTy, WrapperFn, FuncInfo, Args,
                           CS.getLocStart(), CD->getBody()->getLocStart());

  // Initialize the teams reduction scratchpad.
  const OMPExecutableDirective &TD = *getTeamsDirective(WrapperCGF.CGM, D);
  if (isOpenMPTeamsDirective(TD.getDirectiveKind())) {
    Address LocalAddr = WrapperCGF.GetAddrOfLocalVar(ScratchpadArg);
    LValue ArgLVal = WrapperCGF.MakeAddrLValue(
        LocalAddr, ScratchpadArg->getType(), AlignmentSource::Decl);
    llvm::Value *Scratchpad =
        WrapperCGF.EmitLoadOfScalar(ArgLVal, ScratchpadArg->getLocation());
    WrapperCGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_init_params),
        Scratchpad);
  }

  // Call the base outlined function to execute the target region.
  llvm::SmallVector<llvm::Value *, 4> CallArgs;
  for (FunctionArgList::const_iterator I = Args.begin(),
                                       E = std::prev(Args.end());
       I != E; ++I) {
    auto *Arg = *I;
    Address LocalAddr = WrapperCGF.GetAddrOfLocalVar(Arg);
    LValue ArgLVal = WrapperCGF.MakeAddrLValue(LocalAddr, Arg->getType(),
                                               AlignmentSource::Decl);
    llvm::Value *Val = WrapperCGF.EmitLoadOfScalar(ArgLVal, Arg->getLocation());
    CallArgs.emplace_back(Val);
  }
  CGM.getOpenMPRuntime().emitOutlinedFunctionCall(WrapperCGF, CS.getLocStart(),
                                                  OutlinedFn, CallArgs);
  WrapperCGF.FinishFunction();
  return WrapperFn;
}

namespace {
class ExecutionModeRAII {
private:
  CGOpenMPRuntimeNVPTX::ExecutionMode &CurrMode;
  bool &IsOrphaned;

public:
  ExecutionModeRAII(CGOpenMPRuntimeNVPTX::ExecutionMode &CurrMode,
                    CGOpenMPRuntimeNVPTX::ExecutionMode ThisMode,
                    bool &IsOrphaned, bool ThisOrphaned)
      : CurrMode(CurrMode), IsOrphaned(IsOrphaned) {
    CurrMode = ThisMode;
    IsOrphaned = ThisOrphaned;
  }
  ~ExecutionModeRAII() {
    CurrMode = CGOpenMPRuntimeNVPTX::ExecutionMode::UNKNOWN;
    IsOrphaned = !IsOrphaned;
  }
};
}; // namespace

void CGOpenMPRuntimeNVPTX::emitGenericKernel(const OMPExecutableDirective &D,
                                             const TargetKernelProperties &TP,
                                             StringRef ParentName,
                                             llvm::Function *&OutlinedFn,
                                             llvm::Constant *&OutlinedFnID,
                                             bool IsOffloadEntry,
                                             const RegionCodeGenTy &CodeGen) {
  ExecutionModeRAII ModeRAII(CurrMode,
                             CGOpenMPRuntimeNVPTX::ExecutionMode::GENERIC,
                             IsOrphaned, false);
  EntryFunctionState EST(CGM, TP);
  WorkerFunctionState WST(CGM, TP, D.getLocStart());
  Work.clear();
  WrapperFunctionsMap.clear();

  // Emit target region as a standalone region.
  class NVPTXPrePostActionTy : public PrePostActionTy {
    CGOpenMPRuntimeNVPTX &RT;
    CGOpenMPRuntimeNVPTX::EntryFunctionState &EST;
    CGOpenMPRuntimeNVPTX::WorkerFunctionState &WST;

  public:
    NVPTXPrePostActionTy(CGOpenMPRuntimeNVPTX &RT,
                         CGOpenMPRuntimeNVPTX::EntryFunctionState &EST,
                         CGOpenMPRuntimeNVPTX::WorkerFunctionState &WST)
        : RT(RT), EST(EST), WST(WST) {}
    void Enter(CodeGenFunction &CGF) override {
      RT.emitGenericEntryHeader(CGF, EST, WST);
    }
    void Exit(CodeGenFunction &CGF) override {
      RT.emitGenericEntryFooter(CGF, EST);
    }
  } Action(*this, EST, WST);
  CodeGen.setAction(Action);

  emitTargetOutlinedFunctionHelper(D, ParentName, OutlinedFn, OutlinedFnID,
                                   IsOffloadEntry, CodeGen,
                                   /* CaptureLevel = */ 1);

  // Create the worker function
  emitWorkerFunction(WST);

  // Now change the name of the worker function to correspond to this target
  // region's entry function.
  WST.WorkerFn->setName(OutlinedFn->getName() + "_worker");

  return;
}

void CGOpenMPRuntimeNVPTX::emitSPMDKernel(const OMPExecutableDirective &D,
                                          const TargetKernelProperties &TP,
                                          StringRef ParentName,
                                          llvm::Function *&OutlinedFn,
                                          llvm::Constant *&OutlinedFnID,
                                          bool IsOffloadEntry,
                                          const RegionCodeGenTy &CodeGen) {
  ExecutionModeRAII ModeRAII(
      CurrMode, CGOpenMPRuntimeNVPTX::ExecutionMode::SPMD, IsOrphaned, false);
  EntryFunctionState EST(CGM, TP);

  // Emit target region as a standalone region.
  class NVPTXPrePostActionTy : public PrePostActionTy {
    CGOpenMPRuntimeNVPTX &RT;
    CGOpenMPRuntimeNVPTX::EntryFunctionState &EST;
    const OMPExecutableDirective &D;

  public:
    NVPTXPrePostActionTy(CGOpenMPRuntimeNVPTX &RT,
                         CGOpenMPRuntimeNVPTX::EntryFunctionState &EST,
                         const OMPExecutableDirective &D)
        : RT(RT), EST(EST), D(D) {}
    void Enter(CodeGenFunction &CGF) override {
      RT.emitSPMDEntryHeader(CGF, EST, D);
    }
    void Exit(CodeGenFunction &CGF) override {
      RT.emitSPMDEntryFooter(CGF, EST);
    }
  } Action(*this, EST, D);
  CodeGen.setAction(Action);
  emitTargetOutlinedFunctionHelper(D, ParentName, OutlinedFn, OutlinedFnID,
                                   IsOffloadEntry, CodeGen,
                                   /* CaptureLevel = */ 1);
  return;
}

namespace {
class OMPRuntimeAvailabilityRAII {
private:
  bool &IsOMPRuntimeInitialized;
  bool Prev;

public:
  OMPRuntimeAvailabilityRAII(bool &IsOMPRuntimeInitialized,
                             bool TargetRequiresOMPRuntime)
      : IsOMPRuntimeInitialized(IsOMPRuntimeInitialized),
        Prev(IsOMPRuntimeInitialized) {
    IsOMPRuntimeInitialized = TargetRequiresOMPRuntime;
  }
  ~OMPRuntimeAvailabilityRAII() { IsOMPRuntimeInitialized = Prev; }
};
} // namespace

void CGOpenMPRuntimeNVPTX::emitTargetOutlinedFunction(
    const OMPExecutableDirective &D, StringRef ParentName,
    llvm::Function *&OutlinedFn, llvm::Constant *&OutlinedFnID,
    bool IsOffloadEntry, const RegionCodeGenTy &CodeGen,
    unsigned CaptureLevel) {
  if (!IsOffloadEntry) // Nothing to do.
    return;

  assert(!ParentName.empty() && "Invalid target region parent name!");

  TargetKernelProperties TP(CGM, D);

  OMPRuntimeAvailabilityRAII RA(IsOMPRuntimeInitialized,
                                TP.requiresOMPRuntime());
  CGOpenMPRuntimeNVPTX::ExecutionMode Mode = TP.getExecutionMode();
  switch (Mode) {
  case CGOpenMPRuntimeNVPTX::ExecutionMode::GENERIC:
    emitGenericKernel(D, TP, ParentName, OutlinedFn, OutlinedFnID,
                      IsOffloadEntry, CodeGen);
    break;
  case CGOpenMPRuntimeNVPTX::ExecutionMode::SPMD:
    emitSPMDKernel(D, TP, ParentName, OutlinedFn, OutlinedFnID, IsOffloadEntry,
                   CodeGen);
    break;
  default:
    llvm_unreachable(
        "Unknown programming model for OpenMP directive on NVPTX target.");
  }

  SetTargetKernelProperties(CGM, OutlinedFn->getName(), Mode,
                            TP.getReductionVariableCount(),
                            TP.getReductionSizeInBytes());

  CGM.getContext().getDiagnostics().Report(
      D.getLocStart(),
      diag::remark_fe_backend_optimization_remark_analysis_target)
      << (TP.getExecutionMode() == CGOpenMPRuntimeNVPTX::ExecutionMode::GENERIC)
      << TP.requiresOMPRuntime() << (TP.masterSharedDataSize() > 0)
      << TP.masterSharedDataSize() << TP.requiresOMPRuntime()
      << TP.mayContainOrphanedParallel();
  if (TP.requiresOMPRuntime())
    CGM.getContext().getDiagnostics().Report(
        TP.requiresOMPRuntimeReason().Loc,
        diag::remark_fe_backend_optimization_remark_analysis_target_runtime)
        << TP.requiresOMPRuntimeReason().RC;
}

void CGOpenMPRuntimeNVPTX::emitNumThreadsClause(CodeGenFunction &CGF,
                                                llvm::Value *NumThreads,
                                                SourceLocation Loc) {
  // Do nothing in case of SPMD Execution Mode and at level 0.
  if (isSPMDExecutionMode() && InL0())
    return;

  CGOpenMPRuntime::emitNumThreadsClause(CGF, NumThreads, Loc);
}

void CGOpenMPRuntimeNVPTX::emitProcBindClause(CodeGenFunction &CGF,
                                              OpenMPProcBindClauseKind ProcBind,
                                              SourceLocation Loc) {
  // Do nothing in case of SPMD Execution Mode and at level 0.
  if (isSPMDExecutionMode() && InL0())
    return;

  CGOpenMPRuntime::emitProcBindClause(CGF, ProcBind, Loc);
}

void CGOpenMPRuntimeNVPTX::emitForDispatchFinish(CodeGenFunction &CGF,
                                                 const OMPLoopDirective &S,
                                                 SourceLocation Loc,
                                                 unsigned IVSize,
                                                 bool IVSigned) {
  if (!CGF.HaveInsertPoint())
    return;

  //
  // On the NVPTX device a set of threads executing a loop scheduled with a
  // dynamic schedule must complete before starting execution of the next
  // loop with a dynamic schedule.
  //
  // In SPMD mode for directives that combine schedule(dynamic) with
  // dist_schedule(static,chunk) this requirement may not hold as different
  // chunks of the distribute'd loop may be executed simultaneously by
  // parallel threads.
  //
  // NOTES:
  // 1. The NVPTX runtime cannot synchronize threads in dispatch_next
  // because it cannot guarantee convergence.
  // 2. Explicit synchronization is not required in a standard non-SPMD
  // 'parallel for schedule(dynamic)' construct because there is a
  // barrier after the parallel region. In an SPMD construct such as
  // 'target teams distribute parallel for' the barrier after the
  // parallel region is elided.
  // 3. Explicit synchronization is not required for nested parallel
  // constructs since threads are maximally convergent. It is only required for
  // SPMD constructs within a parallel region at the L1 level.
  //

  OpenMPDirectiveKind Kind = S.getDirectiveKind();
  bool DistChunked = false;
  if (auto *C = S.getSingleClause<OMPDistScheduleClause>())
    DistChunked = C->getChunkSize() != nullptr;

  if (isSPMDExecutionMode() && InL1() && DistChunked &&
      (Kind == OMPD_target_teams_distribute_parallel_for ||
       Kind == OMPD_teams_distribute_parallel_for ||
       Kind == OMPD_target_teams_distribute_parallel_for_simd ||
       Kind == OMPD_teams_distribute_parallel_for_simd))
    SyncCTAThreads(CGF);
}

namespace {
///
/// FIXME: This is stupid!
/// These class definitions are duplicated from CGOpenMPRuntime.cpp.  They
/// should instead be placed in the header file CGOpenMPRuntime.h and made
/// accessible to CGOpenMPRuntimeNVPTX.cpp.  Otherwise not only do we have
/// to duplicate code, but we have to ensure that both these definitions are
/// always the same.  This is a problem because a CGOpenMPRegionInfo object
/// from CGOpenMPRuntimeNVPTX.cpp is accessed in methods of CGOpenMPRuntime.cpp.
///
/// \brief Base class for handling code generation inside OpenMP regions.
class CGOpenMPRegionInfo : public CodeGenFunction::CGCapturedStmtInfo {
public:
  /// \brief Kinds of OpenMP regions used in codegen.
  enum CGOpenMPRegionKind {
    /// \brief Region with outlined function for standalone 'parallel'
    /// directive.
    ParallelOutlinedRegion,
    /// \brief Region with outlined function for standalone 'simd'
    /// directive.
    SimdOutlinedRegion,
    /// \brief Region with outlined function for standalone 'task' directive.
    TaskOutlinedRegion,
    /// \brief Region for constructs that do not require function outlining,
    /// like 'for', 'sections', 'atomic' etc. directives.
    InlinedRegion,
    /// \brief Region with outlined function for standalone 'target' directive.
    TargetRegion,
  };

  CGOpenMPRegionInfo(const CapturedStmt &CS,
                     const CGOpenMPRegionKind RegionKind,
                     const RegionCodeGenTy &CodeGen, OpenMPDirectiveKind Kind,
                     bool HasCancel)
      : CGCapturedStmtInfo(CS, CR_OpenMP), RegionKind(RegionKind),
        CodeGen(CodeGen), Kind(Kind), HasCancel(HasCancel) {}

  CGOpenMPRegionInfo(const CGOpenMPRegionKind RegionKind,
                     const RegionCodeGenTy &CodeGen, OpenMPDirectiveKind Kind,
                     bool HasCancel)
      : CGCapturedStmtInfo(CR_OpenMP), RegionKind(RegionKind), CodeGen(CodeGen),
        Kind(Kind), HasCancel(HasCancel) {}

  /// \brief Get a variable or parameter for storing the lane id
  /// inside OpenMP construct.
  virtual const VarDecl *getLaneIDVariable() const { return nullptr; }

  /// \brief Get a variable or parameter for storing the number of lanes
  /// inside OpenMP construct.
  virtual const VarDecl *getNumLanesVariable() const { return nullptr; }

  /// \brief Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  virtual const VarDecl *getThreadIDVariable() const = 0;

  /// \brief Emit the captured statement body.
  void EmitBody(CodeGenFunction &CGF, const Stmt *S) override;

  /// \brief Get an LValue for the current ThreadID variable.
  /// \return LValue for thread id variable. This LValue always has type int32*.
  virtual LValue getThreadIDVariableLValue(CodeGenFunction &CGF);

  /// \brief Get an LValue for the current LaneID variable.
  /// \return LValue for lane id variable. This LValue always has type int32*.
  virtual LValue getLaneIDVariableLValue(CodeGenFunction &CGF);

  /// \brief Get an LValue for the current NumLanes variable.
  /// \return LValue for num lanes variable. This LValue always has type int32*.
  virtual LValue getNumLanesVariableLValue(CodeGenFunction &CGF);

  virtual void emitUntiedSwitch(CodeGenFunction & /*CGF*/) {}

  CGOpenMPRegionKind getRegionKind() const { return RegionKind; }

  OpenMPDirectiveKind getDirectiveKind() const { return Kind; }

  bool hasCancel() const { return HasCancel; }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return Info->getKind() == CR_OpenMP;
  }

  ~CGOpenMPRegionInfo() override = default;

protected:
  CGOpenMPRegionKind RegionKind;
  RegionCodeGenTy CodeGen;
  OpenMPDirectiveKind Kind;
  bool HasCancel;
};

/// \brief API for captured statement code generation in OpenMP constructs.
class CGOpenMPOutlinedRegionInfo final : public CGOpenMPRegionInfo {
public:
  CGOpenMPOutlinedRegionInfo(const CapturedStmt &CS, const VarDecl *ThreadIDVar,
                             const RegionCodeGenTy &CodeGen,
                             OpenMPDirectiveKind Kind, bool HasCancel)
      : CGOpenMPRegionInfo(CS, ParallelOutlinedRegion, CodeGen, Kind,
                           HasCancel),
        ThreadIDVar(ThreadIDVar) {
    assert(ThreadIDVar != nullptr && "No ThreadID in OpenMP region.");
  }

  /// \brief Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  const VarDecl *getThreadIDVariable() const override { return ThreadIDVar; }

  /// \brief Get the name of the capture helper.
  StringRef getHelperName() const override { return ".omp_outlined."; }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return CGOpenMPRegionInfo::classof(Info) &&
           cast<CGOpenMPRegionInfo>(Info)->getRegionKind() ==
               ParallelOutlinedRegion;
  }

private:
  /// \brief A variable or parameter storing global thread id for OpenMP
  /// constructs.
  const VarDecl *ThreadIDVar;
};

/// \brief API for captured statement code generation in OpenMP constructs.
class CGOpenMPSimdOutlinedRegionInfo : public CGOpenMPRegionInfo {
public:
  CGOpenMPSimdOutlinedRegionInfo(const CapturedStmt &CS,
                                 const VarDecl *LaneIDVar,
                                 const VarDecl *NumLanesVar,
                                 const RegionCodeGenTy &CodeGen,
                                 OpenMPDirectiveKind Kind)
      : CGOpenMPRegionInfo(CS, SimdOutlinedRegion, CodeGen, Kind, false),
        LaneIDVar(LaneIDVar), NumLanesVar(NumLanesVar) {
    assert(LaneIDVar != nullptr && "No LaneID in OpenMP region.");
    assert(NumLanesVar != nullptr && "No # Lanes in OpenMP region.");
  }

  /// \brief Get a variable or parameter for storing the lane id
  /// inside OpenMP construct.
  const VarDecl *getLaneIDVariable() const override { return LaneIDVar; }

  /// \brief Get a variable or parameter for storing the number of lanes
  /// inside OpenMP construct.
  const VarDecl *getNumLanesVariable() const override { return NumLanesVar; }

  /// \brief This is unused for simd regions.
  const VarDecl *getThreadIDVariable() const override { return nullptr; }

  /// \brief Get the name of the capture helper.
  StringRef getHelperName() const override { return ".omp_simd_outlined."; }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return CGOpenMPRegionInfo::classof(Info) &&
           cast<CGOpenMPRegionInfo>(Info)->getRegionKind() ==
               SimdOutlinedRegion;
  }

private:
  /// \brief A variable or parameter storing the lane id for OpenMP
  /// constructs.
  const VarDecl *LaneIDVar;
  /// \brief A variable or parameter storing the number of lanes for OpenMP
  /// constructs.
  const VarDecl *NumLanesVar;
};
}

LValue CGOpenMPRegionInfo::getThreadIDVariableLValue(CodeGenFunction &CGF) {
  return CGF.EmitLoadOfPointerLValue(
      CGF.GetAddrOfLocalVar(getThreadIDVariable()),
      getThreadIDVariable()->getType()->castAs<PointerType>());
}

LValue CGOpenMPRegionInfo::getLaneIDVariableLValue(CodeGenFunction &CGF) {
  return CGF.EmitLoadOfPointerLValue(
      CGF.GetAddrOfLocalVar(getLaneIDVariable()),
      getLaneIDVariable()->getType()->castAs<PointerType>());
}

LValue CGOpenMPRegionInfo::getNumLanesVariableLValue(CodeGenFunction &CGF) {
  return CGF.EmitLoadOfPointerLValue(
      CGF.GetAddrOfLocalVar(getNumLanesVariable()),
      getNumLanesVariable()->getType()->castAs<PointerType>());
}

/// Run the provided function with the shared loop bounds of a loop directive.
static void DoOnSharedLoopBounds(
    const OMPExecutableDirective &D,
    const llvm::function_ref<void(const VarDecl *, const VarDecl *)> &Exec) {
  // Is this a loop directive?
  // if (auto *LDir = dyn_cast<OMPLoopDirective>(&D)) {
  if (isOpenMPLoopBoundSharingDirective(D.getDirectiveKind())) {
    auto *LDir = dyn_cast<OMPLoopDirective>(&D);
    // Do the bounds of the associated loop need to be shared? This check is the
    // same as checking the existence of an expression that refers to a previous
    // (enclosing) loop.
    if (LDir->getPrevLowerBoundVariable()) {
      const VarDecl *LB = cast<VarDecl>(
          cast<DeclRefExpr>(LDir->getLowerBoundVariable())->getDecl());
      const VarDecl *UB = cast<VarDecl>(
          cast<DeclRefExpr>(LDir->getUpperBoundVariable())->getDecl());
      Exec(LB, UB);
    }
  }
}

void CGOpenMPRegionInfo::EmitBody(CodeGenFunction &CGF, const Stmt * /*S*/) {
  if (!CGF.HaveInsertPoint())
    return;
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CGF.EHStack.pushTerminate();
  {
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    CodeGen(CGF);
  }
  CGF.EHStack.popTerminate();
}

namespace {
class ParallelNestingLevelRAII {
private:
  unsigned &ParallelNestingLevel;
  unsigned Increment;

public:
  // If in Simd we increase the parallelism level by a bunch to make sure all
  // the Simd regions nested are implemented in a sequential way.
  ParallelNestingLevelRAII(unsigned &ParallelNestingLevel, bool IsSimd = false)
      : ParallelNestingLevel(ParallelNestingLevel), Increment(IsSimd ? 10 : 1) {
    ParallelNestingLevel += Increment;
  }
  ~ParallelNestingLevelRAII() { ParallelNestingLevel -= Increment; }
};
} // namespace

llvm::Value *CGOpenMPRuntimeNVPTX::emitTeamsOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen,
    unsigned CaptureLevel, unsigned ImplicitParamStop) {
  assert(ThreadIDVar->getType()->isPointerType() &&
         "thread id variable must be of type kmp_int32 *");

  if (isSPMDExecutionMode())
    // Create an outlined function if in SPMD mode.
    return CGOpenMPRuntime::emitTeamsOutlinedFunction(
        D, ThreadIDVar, InnermostKind, CodeGen, CaptureLevel,
        ImplicitParamStop);

  // No outlining required for the other teams constructs
  // such as: teams and target teams
  // FIXME: We would like to outline for all teams directive but currently
  // outlining for non SPMD teams directives crashes data sharing.

  return nullptr;
}

llvm::Value *CGOpenMPRuntimeNVPTX::emitParallelOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen,
    unsigned CaptureLevel, unsigned ImplicitParamStop) {
  assert(ThreadIDVar->getType()->isPointerType() &&
         "thread id variable must be of type kmp_int32 *");

  if (isSPMDExecutionMode() && InL0()) {
    // Create an outlined function if in SPMD mode and executing the 'parallel'
    // in the combined directives 'target parallel', 'target parallel for',
    // 'target teams distribute parallel for', or 'teams distribute parallel
    // for'.
    ParallelNestingLevelRAII NestingRAII(ParallelNestingLevel);
    return CGOpenMPRuntime::emitParallelOutlinedFunction(
        D, ThreadIDVar, InnermostKind, CodeGen, CaptureLevel,
        ImplicitParamStop);
  } else {
    // Call to a parallel that is not combined with a teams or target
    // directive (non SPMD).
    // This could also be a nested 'parallel' in an SPMD region.
    const auto *CS = cast<CapturedStmt>(D.getAssociatedStmt());
    if (D.hasClausesOfKind<OMPDependClause>() &&
        isOpenMPTargetExecutionDirective(D.getDirectiveKind()))
      CS = cast<CapturedStmt>(CS->getCapturedStmt());
    CodeGenFunction CGF(CGM, true);
    bool HasCancel = false;
    if (auto *OPD = dyn_cast<OMPParallelDirective>(&D))
      HasCancel = OPD->hasCancel();
    else if (auto *OPSD = dyn_cast<OMPParallelSectionsDirective>(&D))
      HasCancel = OPSD->hasCancel();
    else if (auto *OPFD = dyn_cast<OMPParallelForDirective>(&D))
      HasCancel = OPFD->hasCancel();

    // Save the current parallel context because it may be overwritten by the
    // innermost regions.
    const Decl *CurrentContext = CurrentParallelContext;

    CGOpenMPOutlinedRegionInfo CGInfo(*CS, ThreadIDVar, CodeGen, InnermostKind,
                                      HasCancel);
    CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);
    llvm::Function *OutlinedFun = nullptr;
    {
      ParallelNestingLevelRAII NestingRAII(ParallelNestingLevel);
      // The outlined function takes as arguments the global_tid, bound_tid,
      // and a capture structure created from the captured variables.
      OutlinedFun = CGF.GenerateOpenMPCapturedStmtFunction(
          *CS, /*UseCapturedArgumentsOnly=*/false, CaptureLevel);
    }
    auto *WrapperFun =
        createDataSharingParallelWrapper(*OutlinedFun, D, CurrentContext);
    WrapperFunctionsMap[OutlinedFun] = WrapperFun;
    return OutlinedFun;
  }
}

llvm::Value *CGOpenMPRuntimeNVPTX::emitSimdOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *LaneIDVar,
    const VarDecl *NumLanesVar, OpenMPDirectiveKind InnermostKind,
    const RegionCodeGenTy &CodeGen) {
  llvm::Function *OutlinedFun = nullptr;

  const auto *CS = cast<CapturedStmt>(D.getAssociatedStmt());
  if (D.hasClausesOfKind<OMPDependClause>() &&
      isOpenMPTargetExecutionDirective(D.getDirectiveKind()))
    CS = cast<CapturedStmt>(CS->getCapturedStmt());

  // Save the current parallel context because it may be overwritten by the
  // innermost regions.
  const Decl *CurrentContext = CurrentParallelContext;

  CodeGenFunction CGF(CGM, true);
  CGOpenMPSimdOutlinedRegionInfo CGInfo(*CS, LaneIDVar, NumLanesVar, CodeGen,
                                        InnermostKind);
  CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);
  {
    ParallelNestingLevelRAII NestingRAII(ParallelNestingLevel, /*IsSimd=*/true);
    OutlinedFun = CGF.GenerateOpenMPCapturedStmtFunction(*CS);
  }

  auto *WrapperFun = createDataSharingParallelWrapper(
      *OutlinedFun, D, CurrentContext, /*IsSimd=*/true);
  WrapperFunctionsMap[OutlinedFun] = WrapperFun;
  return OutlinedFun;
}

bool CGOpenMPRuntimeNVPTX::InL0() {
  return !IsOrphaned && ParallelNestingLevel == 0;
}

bool CGOpenMPRuntimeNVPTX::InL1() {
  return !IsOrphaned && ParallelNestingLevel == 1;
}

bool CGOpenMPRuntimeNVPTX::InL1Plus() {
  return !IsOrphaned && ParallelNestingLevel >= 1;
}

bool CGOpenMPRuntimeNVPTX::isSPMDExecutionMode() const {
  return CurrMode == CGOpenMPRuntimeNVPTX::ExecutionMode::SPMD;
}

bool CGOpenMPRuntimeNVPTX::isOMPRuntimeInitialized() const {
  return IsOMPRuntimeInitialized;
}

void CGOpenMPRuntimeNVPTX::registerCtorDtorEntry(
    unsigned DeviceID, unsigned FileID, StringRef RegionName, unsigned Line,
    llvm::Function *Fn, bool IsDtor) {
  // On top of the default registration we create a new global to force the
  // region to be executed as SPMD.
  SetTargetKernelProperties(CGM, Fn->getName(), SPMD,
                            /*ReductionVariableCount=*/0,
                            /*ReductionSizeInBytes=*/0);

  CGOpenMPRuntime::registerCtorDtorEntry(DeviceID, FileID, RegionName, Line, Fn,
                                         IsDtor);
}

bool CGOpenMPRuntimeNVPTX::IndeterminateLevel() { return IsOrphaned; }

// \brief Obtain the data sharing info for the current context.
const CGOpenMPRuntimeNVPTX::DataSharingInfo &
CGOpenMPRuntimeNVPTX::getDataSharingInfo(const Decl *Context) {
  assert(Context &&
         "A parallel region is expected to be enclosed in a context.");

  auto It = DataSharingInfoMap.find(Context);
  assert(It != DataSharingInfoMap.end() && "Data sharing info does not exist.");
  return It->second;
}

void CGOpenMPRuntimeNVPTX::createDataSharingInfo(CodeGenFunction &CGF) {
  auto &Context = CGF.CurCodeDecl;
  assert(Context &&
         "A parallel region is expected to be enclosed in a context.");

  ASTContext &C = CGM.getContext();

  if (DataSharingInfoMap.find(Context) != DataSharingInfoMap.end())
    return;

  auto &Info = DataSharingInfoMap[Context];

  // Get the body of the region. The region context is either a function or a
  // captured declaration.
  const Stmt *Body;
  if (auto *D = dyn_cast<CapturedDecl>(Context))
    Body = D->getBody();
  else
    Body = cast<FunctionDecl>(Context)->getBody();

  // Track if in this region one has to share

  // Find all the captures in all enclosed regions and obtain their captured
  // statements.
  SmallVector<const OMPExecutableDirective *, 8> CapturedDirs;
  SmallVector<const Stmt *, 64> WorkList;
  WorkList.push_back(Body);
  while (!WorkList.empty()) {
    const Stmt *CurStmt = WorkList.pop_back_val();
    if (!CurStmt)
      continue;

    // Is this a parallel region.
    if (auto *Dir = dyn_cast<OMPExecutableDirective>(CurStmt)) {
      if (isOpenMPParallelDirective(Dir->getDirectiveKind()) ||
          isOpenMPSimdDirective(Dir->getDirectiveKind())) {
        CapturedDirs.push_back(Dir);
      } else {
        if (Dir->hasAssociatedStmt()) {
          // Look into the associated statement of OpenMP directives.
          const CapturedStmt &CS =
              *cast<CapturedStmt>(Dir->getAssociatedStmt());
          CurStmt = CS.getCapturedStmt();

          WorkList.push_back(CurStmt);
        }
      }

      continue;
    }

    // Keep looking for other regions.
    WorkList.append(CurStmt->child_begin(), CurStmt->child_end());
  }

  assert(!CapturedDirs.empty() && "Expecting at least one parallel region!");

  // Scan the captured statements and generate a record to contain all the data
  // to be shared. Make sure we do not share the same thing twice.
  auto *SharedMasterRD =
      C.buildImplicitRecord("__openmp_nvptx_data_sharing_master_record");
  auto *SharedWarpRD =
      C.buildImplicitRecord("__openmp_nvptx_data_sharing_warp_record");
  SharedMasterRD->startDefinition();
  SharedWarpRD->startDefinition();

  llvm::SmallSet<const VarDecl *, 32> AlreadySharedDecls;
  ASTContext &Ctx = CGF.getContext();
  for (auto *Dir : CapturedDirs) {
    const auto *CS = cast<CapturedStmt>(Dir->getAssociatedStmt());
    if (Dir->hasClausesOfKind<OMPDependClause>() &&
        isOpenMPTargetExecutionDirective(Dir->getDirectiveKind()))
      CS = cast<CapturedStmt>(CS->getCapturedStmt());
    const RecordDecl *RD = CS->getCapturedRecordDecl();
    auto CurField = RD->field_begin();
    auto CurCap = CS->capture_begin();
    for (CapturedStmt::const_capture_init_iterator I = CS->capture_init_begin(),
                                                   E = CS->capture_init_end();
         I != E; ++I, ++CurField, ++CurCap) {

      const VarDecl *CurVD = nullptr;
      QualType ElemTy;
      if (*I)
        ElemTy = (*I)->getType();

      // Track the data sharing type.
      DataSharingInfo::DataSharingType DST = DataSharingInfo::DST_Val;

      if (CurCap->capturesThis()) {
        // We use null to indicate 'this'.
        CurVD = nullptr;
      } else {
        // Get the variable that is initializing the capture.
        if (CurField->hasCapturedVLAType()) {
          auto VAT = CurField->getCapturedVLAType();
          ElemTy = Ctx.getSizeType();
          CurVD = ImplicitParamDecl::Create(
              Ctx, /*DC=*/nullptr, CurField->getLocation(), /*Id=*/nullptr,
              ElemTy, ImplicitParamDecl::Other);
          CGF.EmitVarDecl(*CurVD);
          CGF.Builder.CreateAlignedStore(CGF.Builder.CreateIntCast(
              CGF.VLASizeMap[VAT->getSizeExpr()], CGM.SizeTy, false),
              CGF.GetAddrOfLocalVar(CurVD).getPointer(),
              CGF.getPointerAlign());
          Info.addVLADecl(VAT->getSizeExpr(), CurVD);
        } else
          CurVD = CurCap->getCapturedVar();

        // If this is an OpenMP capture declaration, we need to look at the
        // original declaration.
        const VarDecl *OrigVD = CurVD;
        if (auto *OD = dyn_cast<OMPCapturedExprDecl>(OrigVD))
          OrigVD = cast<VarDecl>(
              cast<DeclRefExpr>(OD->getInit()->IgnoreImpCasts())->getDecl());

        // If the variable does not have local storage it is always a reference.
        if (!OrigVD->hasLocalStorage())
          DST = DataSharingInfo::DST_Ref;
        else {
          // If we have an alloca for this variable, then we need to share the
          // storage too, not only the reference.
          auto *Val = cast<llvm::Instruction>(
              CGF.GetAddrOfLocalVar(OrigVD).getPointer());
          if (isa<llvm::LoadInst>(Val))
            DST = DataSharingInfo::DST_Ref;
          // If the variable is a bitcast, it is being encoded in a pointer
          // and should be treated as such.
          else if (isa<llvm::BitCastInst>(Val))
            DST = DataSharingInfo::DST_Cast;
          // If the variable is a reference, we also share it as is,
          // i.e., consider it a reference to something that can be shared.
          else if (OrigVD->getType()->isReferenceType())
            DST = DataSharingInfo::DST_Ref;
        }
      }

      // Do not insert the same declaration twice.
      if (AlreadySharedDecls.count(CurVD))
        continue;

      AlreadySharedDecls.insert(CurVD);
      Info.add(CurVD, DST);

      if (DST == DataSharingInfo::DST_Ref)
        ElemTy = C.getPointerType(ElemTy);

      addFieldToRecordDecl(C, SharedMasterRD, ElemTy);
      llvm::APInt NumElems(C.getTypeSize(C.getUIntPtrType()),
                           DS_Max_Worker_Warp_Size);
      auto QTy = C.getConstantArrayType(ElemTy, NumElems, ArrayType::Normal,
                                        /*IndexTypeQuals=*/0);
      addFieldToRecordDecl(C, SharedWarpRD, QTy);
    }

    // Add loop bounds if required.
    DoOnSharedLoopBounds(*Dir, [&AlreadySharedDecls, &C, &Info, &SharedMasterRD,
                                &SharedWarpRD, &Dir,
                                &CGF](const VarDecl *LB, const VarDecl *UB) {
      // Do not insert the same declaration twice.
      if (AlreadySharedDecls.count(LB))
        return;

      // We assume that if the lower bound is not to be shared, the upper
      // bound is not shared as well.
      assert(!AlreadySharedDecls.count(UB) &&
             "Not expecting shared upper bound.");

      QualType ElemTy = LB->getType();

      // Bounds are shared by value.
      Info.add(LB, DataSharingInfo::DST_Val);
      Info.add(UB, DataSharingInfo::DST_Val);
      addFieldToRecordDecl(C, SharedMasterRD, ElemTy);
      addFieldToRecordDecl(C, SharedMasterRD, ElemTy);

      llvm::APInt NumElems(C.getTypeSize(C.getUIntPtrType()),
                           DS_Max_Worker_Warp_Size);
      auto QTy = C.getConstantArrayType(ElemTy, NumElems, ArrayType::Normal,
                                        /*IndexTypeQuals=*/0);
      addFieldToRecordDecl(C, SharedWarpRD, QTy);
      addFieldToRecordDecl(C, SharedWarpRD, QTy);

      // Emit the preinits to make sure the initializers are properly
      // emitted.
      // FIXME: This is a hack - it won't work if declarations being shared
      // appear after the first parallel region.
      const OMPLoopDirective *L = cast<OMPLoopDirective>(Dir);
      if (auto *PreInits = cast_or_null<DeclStmt>(L->getPreInits()))
        for (const auto *I : PreInits->decls()) {
          CGF.EmitOMPHelperVar(cast<VarDecl>(I));
        }
    });
  }

  SharedMasterRD->completeDefinition();
  SharedWarpRD->completeDefinition();
  Info.MasterRecordType = C.getRecordType(SharedMasterRD);
  Info.WorkerWarpRecordType = C.getRecordType(SharedWarpRD);

  return;
}

// Cast an address from the requested type to uintptr in such a way that it can
// be loaded under the new type. If the provided address refers to a pointer
// don't do anything an return the address as is.
static LValue castValueToUintptr(CodeGenFunction &CGF, QualType SrcType,
                                 StringRef Name, LValue AddrLV,
                                 SourceLocation Loc) {

  // If the value is a pointer we don't have to do anything.
  if (SrcType->isAnyPointerType())
    return AddrLV;

  ASTContext &Ctx = CGF.getContext();

  // Value to be converted.
  auto *Val = CGF.EmitLoadOfScalar(AddrLV, Loc);

  // Create a temporary variable of type uintptr to make the conversion and cast
  // address to the desired type.
  auto CastAddr =
      CGF.CreateMemTemp(Ctx.getUIntPtrType(), Twine(Name) + ".casted");
  auto *CastAddrConv =
      CGF.EmitScalarConversion(CastAddr.getPointer(), Ctx.getUIntPtrType(),
                               Ctx.getPointerType(SrcType), Loc);
  auto CastAddrConvLV = CGF.MakeNaturalAlignAddrLValue(CastAddrConv, SrcType);

  // Save the value in the temporary variable.
  CGF.EmitStoreOfScalar(Val, CastAddrConvLV);

  // Return the temporary variable address.
  return CGF.MakeAddrLValue(CastAddr, Ctx.getUIntPtrType());
}

void CGOpenMPRuntimeNVPTX::createDataSharingPerFunctionInfrastructure(
    CodeGenFunction &EnclosingCGF) {
  const Decl *CD = EnclosingCGF.CurCodeDecl;
  auto &Ctx = CGM.getContext();

  assert(CD && "Function does not have a context associated!");

  // Create the data sharing information.
  auto &DSI = getDataSharingInfo(CD);

  // If there is nothing being captured in the parallel regions, we do not need
  // to do anything.
  if (DSI.CapturesValues.empty())
    return;

  auto &EnclosingFuncInfo = DataSharingFunctionInfoMap[EnclosingCGF.CurFn];

  // If we already have a data sharing initializer of this function, don't need
  // to create a new one.
  if (EnclosingFuncInfo.InitializationFunction)
    return;

  auto IsEntryPoint = EnclosingFuncInfo.IsEntryPoint;

  // Create function to do the initialization. The first four arguments are the
  // slot/stack/frame saved addresses and then we have pairs of pointers to the
  // shared address and each declaration to be shared.
  // FunctionArgList ArgList;
  SmallVector<ImplicitParamDecl, 4> ArgImplDecls;

  // Create the variables to save the slot, stack, frame and active threads.
  QualType SlotPtrTy = Ctx.getPointerType(getDataSharingSlotQty());
  QualType Int32QTy =
      Ctx.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/false);
  ArgImplDecls.push_back(ImplicitParamDecl(
      Ctx, /*DC=*/nullptr, CD->getLocation(),
      &Ctx.Idents.get("data_share_saved_slot"), Ctx.getPointerType(SlotPtrTy),
      ImplicitParamDecl::Other));
  ArgImplDecls.push_back(ImplicitParamDecl(
      Ctx, /*DC=*/nullptr, CD->getLocation(),
      &Ctx.Idents.get("data_share_saved_stack"),
      Ctx.getPointerType(Ctx.VoidPtrTy), ImplicitParamDecl::Other));
  ArgImplDecls.push_back(ImplicitParamDecl(
      Ctx, /*DC=*/nullptr, CD->getLocation(),
      &Ctx.Idents.get("data_share_saved_frame"),
      Ctx.getPointerType(Ctx.VoidPtrTy), ImplicitParamDecl::Other));
  ArgImplDecls.push_back(ImplicitParamDecl(
      Ctx, /*DC=*/nullptr, CD->getLocation(),
      &Ctx.Idents.get("data_share_active_threads"),
      Ctx.getPointerType(Int32QTy), ImplicitParamDecl::Other));

  auto *MasterRD = DSI.MasterRecordType->getAs<RecordType>()->getDecl();
  auto CapturesIt = DSI.CapturesValues.begin();
  for (auto *F : MasterRD->fields()) {
    QualType ArgTy = F->getType();
    if (ArgTy->isVariablyModifiedType()) {
      bool IsReference = ArgTy->isLValueReferenceType();
      ArgTy = Ctx.getCanonicalParamType(ArgTy.getNonReferenceType());
      if (IsReference && !ArgTy->isPointerType()) {
        ArgTy = Ctx.getLValueReferenceType(ArgTy, /*SpelledAsLValue=*/false);
      }
    }

    // If this is not a reference the right address type is the pointer type of
    // the type that is the record.
    if (CapturesIt->second != DataSharingInfo::DST_Ref)
      ArgTy = Ctx.getPointerType(ArgTy);

    StringRef BaseName =
        CapturesIt->first ? CapturesIt->first->getName() : "this";

    // If this is not a reference, we need to return by reference the new
    // address to be replaced.
    if (CapturesIt->second != DataSharingInfo::DST_Ref) {
      std::string Name = BaseName;
      Name += ".addr";
      auto &NameID = Ctx.Idents.get(Name);
      ImplicitParamDecl D(Ctx, /*DC=*/nullptr, CD->getLocation(), &NameID,
                          Ctx.getPointerType(ArgTy), ImplicitParamDecl::Other);
      ArgImplDecls.push_back(D);
    }

    std::string NameOrig = BaseName;
    NameOrig += ".orig";
    auto &NameOrigID = Ctx.Idents.get(NameOrig);
    ImplicitParamDecl OrigD(Ctx, /*DC=*/nullptr, CD->getLocation(), &NameOrigID,
                            ArgTy, ImplicitParamDecl::Other);
    ArgImplDecls.push_back(OrigD);

    ++CapturesIt;
  }

  FunctionArgList ArgList;
  int count = 0;
  for (auto &I : ArgImplDecls) {
    ArgList.push_back(&I);
    count++;
  }

  auto &CGFI =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(Ctx.VoidTy, ArgList);
  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      EnclosingCGF.CurFn->getName() + ".data_share", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, Fn, CGFI);
  Fn->setLinkage(llvm::GlobalValue::InternalLinkage);

  CodeGenFunction CGF(CGM, /*suppressNewContext=*/true);
  CGF.disableDebugInfo();
  CGF.StartFunction(GlobalDecl(), Ctx.VoidTy, Fn, CGFI, ArgList,
                    CD->getLocation(), CD->getLocation());

  // If this is an entry point, all the threads except the master should skip
  // this.
  auto *ExitBB = CGF.createBasicBlock(".exit");
  if (IsEntryPoint) {
    auto *MasterBB = CGF.createBasicBlock(".master");
    auto *Cond =
        CGF.Builder.CreateICmpEQ(GetMasterThreadID(CGF), GetNVPTXThreadID(CGF));
    CGF.Builder.CreateCondBr(Cond, MasterBB, ExitBB);
    CGF.EmitBlock(MasterBB);
  }

  // Create the variables to save the slot, stack, frame and active threads.
  auto ArgsIt = ArgList.begin();
  auto SavedSlotAddr =
      CGF.EmitLoadOfPointer(CGF.GetAddrOfLocalVar(*ArgsIt),
                            (*ArgsIt)->getType()->getAs<PointerType>());
  ++ArgsIt;
  auto SavedStackAddr =
      CGF.EmitLoadOfPointer(CGF.GetAddrOfLocalVar(*ArgsIt),
                            (*ArgsIt)->getType()->getAs<PointerType>());
  ++ArgsIt;
  auto SavedFrameAddr =
      CGF.EmitLoadOfPointer(CGF.GetAddrOfLocalVar(*ArgsIt),
                            (*ArgsIt)->getType()->getAs<PointerType>());
  ++ArgsIt;
  auto SavedActiveThreadsAddr =
      CGF.EmitLoadOfPointer(CGF.GetAddrOfLocalVar(*ArgsIt),
                            (*ArgsIt)->getType()->getAs<PointerType>());
  ++ArgsIt;

  auto *SavedSlot = SavedSlotAddr.getPointer();
  auto *SavedStack = SavedStackAddr.getPointer();
  auto *SavedFrame = SavedFrameAddr.getPointer();
  auto *SavedActiveThreads = SavedActiveThreadsAddr.getPointer();

  // Get the addresses where each data shared address will be stored.
  SmallVector<Address, 32> NewAddressPtrs;
  SmallVector<Address, 32> OrigAddresses;
  // We iterate two by two.
  for (auto CapturesIt = DSI.CapturesValues.begin();
       // !(CapturesIt == DSI.CapturesValues.end() || ArgsIt == ArgList.end());
       ArgsIt != ArgList.end();
       ++ArgsIt, ++CapturesIt) {
    if (CapturesIt->second != DataSharingInfo::DST_Ref) {
      NewAddressPtrs.push_back(
          CGF.EmitLoadOfPointer(CGF.GetAddrOfLocalVar(*ArgsIt),
                                (*ArgsIt)->getType()->getAs<PointerType>()));
      ++ArgsIt;
    }

    OrigAddresses.push_back(
        CGF.EmitLoadOfPointer(CGF.GetAddrOfLocalVar(*ArgsIt),
                              (*ArgsIt)->getType()->getAs<PointerType>()));
  }

  auto &&L0ParallelGen = [this, &DSI, MasterRD, &Ctx, SavedSlot, SavedStack, CD,
                          SavedFrame, SavedActiveThreads, &NewAddressPtrs,
                          &OrigAddresses](CodeGenFunction &CGF,
                                          PrePostActionTy &) {
    auto &Bld = CGF.Builder;

    // In the Level 0 regions, we use the master record to get the data.
    auto *DataSize = llvm::ConstantInt::get(
        CGM.SizeTy, Ctx.getTypeSizeInChars(DSI.MasterRecordType).getQuantity());
    auto *DefaultDataSize = llvm::ConstantInt::get(CGM.SizeTy, DS_Slot_Size);

    llvm::Value *Args[] = {SavedSlot,
                           SavedStack,
                           SavedFrame,
                           SavedActiveThreads,
                           DataSize,
                           DefaultDataSize,
                           Bld.getInt16(isOMPRuntimeInitialized() ? 1 : 0)};
    auto *DataShareAddr = CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(
            OMPRTL_NVPTX__kmpc_data_sharing_environment_begin),
        Args, "data_share_master_addr");
    auto DataSharePtrQTy = Ctx.getPointerType(DSI.MasterRecordType);
    auto *DataSharePtrTy = CGF.getTypes().ConvertTypeForMem(DataSharePtrQTy);
    auto *CasterDataShareAddr =
        Bld.CreateBitOrPointerCast(DataShareAddr, DataSharePtrTy);

    // For each field, return the address by reference if it is not a reference
    // capture, otherwise copy the original pointer to the shared address space.
    // If it is a cast, we need to copy the pointee into shared memory.
    auto FI = MasterRD->field_begin();
    auto CapturesIt = DSI.CapturesValues.begin();
    auto NewAddressIt = NewAddressPtrs.begin();
    for (unsigned i = 0; i < OrigAddresses.size(); ++i, ++FI, ++CapturesIt) {
      llvm::Value *Idx[] = {Bld.getInt32(0), Bld.getInt32(i)};
      auto *NewAddr = Bld.CreateInBoundsGEP(CasterDataShareAddr, Idx);

      switch (CapturesIt->second) {
      case DataSharingInfo::DST_Ref: {
        auto Addr = CGF.MakeNaturalAlignAddrLValue(NewAddr, FI->getType());
        CGF.EmitStoreOfScalar(OrigAddresses[i].getPointer(), Addr);
      } break;
      case DataSharingInfo::DST_Cast: {
        // Copy the pointee to the new location.
        auto *PointeeVal =
            CGF.EmitLoadOfScalar(OrigAddresses[i], /*Volatiole=*/false,
                                 FI->getType(), CD->getLocation());
        auto NewAddrLVal =
            CGF.MakeNaturalAlignAddrLValue(NewAddr, FI->getType());
        CGF.EmitStoreOfScalar(PointeeVal, NewAddrLVal);
      } // fallthrough.
      case DataSharingInfo::DST_Val: {
        CGF.EmitStoreOfScalar(NewAddr, *NewAddressIt, /*Volatile=*/false,
                              Ctx.getPointerType(FI->getType()));
        ++NewAddressIt;
      } break;
      }
    }
  };
  auto &&L1ParallelGen = [this, &DSI, MasterRD, &Ctx, SavedSlot, SavedStack, CD,
                          SavedFrame, SavedActiveThreads, &NewAddressPtrs,
                          &OrigAddresses](CodeGenFunction &CGF,
                                          PrePostActionTy &) {
    auto &Bld = CGF.Builder;

    // In the Level 1 regions, we use the worker record that has each capture
    // organized as an array.
    auto *DataSize = llvm::ConstantInt::get(
        CGM.SizeTy,
        Ctx.getTypeSizeInChars(DSI.WorkerWarpRecordType).getQuantity());
    auto *DefaultDataSize =
        llvm::ConstantInt::get(CGM.SizeTy, DS_Worker_Warp_Slot_Size);

    llvm::Value *Args[] = {SavedSlot,
                           SavedStack,
                           SavedFrame,
                           SavedActiveThreads,
                           DataSize,
                           DefaultDataSize,
                           /*isOMPRuntimeInitialized=*/Bld.getInt16(1)};
    auto *DataShareAddr = CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(
            OMPRTL_NVPTX__kmpc_data_sharing_environment_begin),
        Args, "data_share_master_addr");
    auto DataSharePtrQTy = Ctx.getPointerType(DSI.WorkerWarpRecordType);
    auto *DataSharePtrTy = CGF.getTypes().ConvertTypeForMem(DataSharePtrQTy);
    auto *CasterDataShareAddr =
        Bld.CreateBitOrPointerCast(DataShareAddr, DataSharePtrTy);

    // Get the threadID in the warp. We have a frame per warp.
    auto *ThreadWarpID = GetNVPTXThreadWarpID(CGF);

    // For each field, generate the shared address and store it in the new
    // addresses array.
    auto FI = MasterRD->field_begin();
    auto CapturesIt = DSI.CapturesValues.begin();
    auto NewAddressIt = NewAddressPtrs.begin();
    for (unsigned i = 0; i < OrigAddresses.size(); ++i, ++FI, ++CapturesIt) {
      llvm::Value *Idx[] = {Bld.getInt32(0), Bld.getInt32(i), ThreadWarpID};
      auto *NewAddr = Bld.CreateInBoundsGEP(CasterDataShareAddr, Idx);

      switch (CapturesIt->second) {
      case DataSharingInfo::DST_Ref: {
        auto Addr = CGF.MakeNaturalAlignAddrLValue(NewAddr, FI->getType());
        CGF.EmitStoreOfScalar(OrigAddresses[i].getPointer(), Addr);
      } break;
      case DataSharingInfo::DST_Cast: {
        // Copy the pointee to the new location.
        auto *PointeeVal =
            CGF.EmitLoadOfScalar(OrigAddresses[i], /*Volatiole=*/false,
                                 FI->getType(), CD->getLocation());
        auto NewAddrLVal =
            CGF.MakeNaturalAlignAddrLValue(NewAddr, FI->getType());
        CGF.EmitStoreOfScalar(PointeeVal, NewAddrLVal);
      } // fallthrough.
      case DataSharingInfo::DST_Val: {
        CGF.EmitStoreOfScalar(NewAddr, *NewAddressIt, /*Volatile=*/false,
                              Ctx.getPointerType(FI->getType()));
        ++NewAddressIt;
      } break;
      }
    }
  };
  auto &&Sequential = [&DSI, &Ctx, MasterRD, &NewAddressPtrs, &OrigAddresses](
                          CodeGenFunction &CGF, PrePostActionTy &) {
    // In the sequential regions, we just use the regular allocas.
    auto FI = MasterRD->field_begin();
    auto CapturesIt = DSI.CapturesValues.begin();
    auto NewAddressIt = NewAddressPtrs.begin();
    for (unsigned i = 0; i < OrigAddresses.size(); ++i, ++FI, ++CapturesIt) {
      // If capturing a reference, the original value will be used.
      if (CapturesIt->second == DataSharingInfo::DST_Ref)
        continue;

      llvm::Value *OriginalVal = OrigAddresses[i].getPointer();
      CGF.EmitStoreOfScalar(OriginalVal, *NewAddressIt,
                            /*Volatile=*/false,
                            Ctx.getPointerType(FI->getType()));
      ++NewAddressIt;
    }
  };

  auto &&ParallelLevelGen = [this, &CGF, CD]() -> llvm::Value * {
    return getParallelLevel(CGF, CD->getLocation());
  };
  emitParallelismLevelCode(CGF, ParallelLevelGen, L0ParallelGen, L1ParallelGen,
                           Sequential);

  // Generate the values to replace.
  auto FI = MasterRD->field_begin();
  for (unsigned i = 0; i < OrigAddresses.size(); ++i, ++FI) {
    llvm::Value *OriginalVal = nullptr;
    if (const VarDecl *VD = DSI.CapturesValues[i].first) {
      if (DSI.isVLADecl(VD)) {
        EnclosingCGF.EmitOMPHelperVar(VD);
        Address OriginalAddr = EnclosingCGF.GetAddrOfLocalVar(VD);
        OriginalVal = OriginalAddr.getPointer();
      } else {
        DeclRefExpr DRE(const_cast<VarDecl *>(VD),
                        /*RefersToEnclosingVariableOrCapture=*/false,
                        VD->getType().getNonReferenceType(), VK_LValue,
                        CD->getLocation());
        Address OriginalAddr = EnclosingCGF.EmitOMPHelperVar(&DRE).getAddress();
        OriginalVal = OriginalAddr.getPointer();
      }
    } else
      OriginalVal = CGF.LoadCXXThis();

    assert(OriginalVal && "Can't obtain value to replace with??");

    EnclosingFuncInfo.ValuesToBeReplaced.push_back(std::make_pair(
        OriginalVal, DSI.CapturesValues[i].second == DataSharingInfo::DST_Ref));
  }

  CGF.EmitBlock(ExitBB);
  CGF.FinishFunction();

  EnclosingFuncInfo.InitializationFunction = CGF.CurFn;
}

// Store the data sharing address of the provided variable (null for 'this').
static void CreateAddressStoreForVariable(
    CodeGenFunction &CGF, const VarDecl *VD, QualType Ty,
    const CGOpenMPRuntimeNVPTX::DataSharingInfo &DSI, llvm::Value *SlotAddr,
    Address StoreAddr, SourceLocation Loc, llvm::Value *LaneID = nullptr) {
  auto &Ctx = CGF.getContext();
  auto &Bld = CGF.Builder;

  unsigned Idx = 0;
  for (; Idx < DSI.CapturesValues.size(); ++Idx)
    if (DSI.CapturesValues[Idx].first == VD)
      break;
  assert(Idx != DSI.CapturesValues.size() && "Capture must exist!");

  llvm::Value *Arg;
  if (LaneID) {
    llvm::Value *Idxs[] = {Bld.getInt32(0), Bld.getInt32(Idx), LaneID};
    Arg = Bld.CreateInBoundsGEP(SlotAddr, Idxs);
  } else {
    llvm::Value *Idxs[] = {Bld.getInt32(0), Bld.getInt32(Idx)};
    Arg = Bld.CreateInBoundsGEP(SlotAddr, Idxs);
  }

  // If what is being shared is the reference, we should load it.
  if (DSI.CapturesValues[Idx].second ==
      CGOpenMPRuntimeNVPTX::DataSharingInfo::DST_Ref) {
    auto Addr = CGF.MakeNaturalAlignAddrLValue(Arg, Ty);
    Arg = CGF.EmitLoadOfScalar(Addr, Loc);
    CGF.EmitStoreOfScalar(Arg, StoreAddr, /*Volatile=*/false, Ty);
  } else
    CGF.EmitStoreOfScalar(Arg, StoreAddr, /*Volatile=*/false,
                          Ctx.getPointerType(Ty));
}

// \brief Create the data sharing arguments and call the parallel outlined
// function.
llvm::Function *CGOpenMPRuntimeNVPTX::createDataSharingParallelWrapper(
    llvm::Function &OutlinedParallelFn, const OMPExecutableDirective &D,
    const Decl *CurrentContext, bool IsSimd) {
  auto &Ctx = CGM.getContext();
  const auto *PCS = cast<CapturedStmt>(D.getAssociatedStmt());
  if (D.hasClausesOfKind<OMPDependClause>() &&
      isOpenMPTargetExecutionDirective(D.getDirectiveKind()))
    PCS = cast<CapturedStmt>(PCS->getCapturedStmt());
  const CapturedStmt &CS = *PCS;

  // Create a function that takes as argument the source lane.
  FunctionArgList WrapperArgs;
  QualType Int16QTy =
      Ctx.getIntTypeForBitwidth(/*DestWidth=*/16, /*Signed=*/false);
  QualType Int32QTy =
      Ctx.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/false);
  QualType Int32PtrQTy = Ctx.getPointerType(Int32QTy);
  ImplicitParamDecl ParallelLevelArg(Ctx, /*DC=*/nullptr, D.getLocStart(),
                                     /*Id=*/nullptr, Int16QTy,
                                     ImplicitParamDecl::Other);
  ImplicitParamDecl WrapperArg(Ctx, /*DC=*/nullptr, D.getLocStart(),
                               /*Id=*/nullptr, Int32QTy,
                               ImplicitParamDecl::Other);
  ImplicitParamDecl WrapperLaneArg(Ctx, /*DC=*/nullptr, D.getLocStart(),
                                   /*Id=*/nullptr, Int32PtrQTy,
                                   ImplicitParamDecl::Other);
  ImplicitParamDecl WrapperNumLanesArg(Ctx, /*DC=*/nullptr, D.getLocStart(),
                                       /*Id=*/nullptr, Int32PtrQTy,
                                       ImplicitParamDecl::Other);
  WrapperArgs.push_back(&ParallelLevelArg);
  WrapperArgs.push_back(&WrapperArg);
  if (IsSimd) {
    WrapperArgs.push_back(&WrapperLaneArg);
    WrapperArgs.push_back(&WrapperNumLanesArg);
  }

  auto &CGFI =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(Ctx.VoidTy, WrapperArgs);

  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      OutlinedParallelFn.getName() + "_wrapper", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, Fn, CGFI);
  Fn->removeFnAttr(llvm::Attribute::NoInline);
  Fn->addFnAttr(llvm::Attribute::AlwaysInline);
  Fn->setLinkage(llvm::GlobalValue::InternalLinkage);

  CodeGenFunction CGF(CGM, /*suppressNewContext=*/true);
  CGF.StartFunction(GlobalDecl(), Ctx.VoidTy, Fn, CGFI, WrapperArgs,
                    D.getLocStart(), D.getLocStart());

  // Get the parallelism level (L0, L1, L2+).
  auto ParallelLevelAddr = CGF.GetAddrOfLocalVar(&ParallelLevelArg);

  // Get the source thread ID, it is the argument of the current function.
  auto SourceLaneIDAddr = CGF.GetAddrOfLocalVar(&WrapperArg);
  auto *SourceLaneID = CGF.EmitLoadOfScalar(
      SourceLaneIDAddr, /*Volatile=*/false, Int32QTy, D.getLocStart());

  // Create temporary variables to contain the new args.
  SmallVector<Address, 32> ArgsAddresses;

  // Get the data sharing information for the context that encloses the current
  // one.
  auto &DSI = getDataSharingInfo(CurrentContext);

  auto *RD = CS.getCapturedRecordDecl();
  auto CurField = RD->field_begin();
  for (CapturedStmt::const_capture_iterator CI = CS.capture_begin(),
                                            CE = CS.capture_end();
       CI != CE; ++CI, ++CurField) {
    QualType ElemTy = CurField->getType();
    StringRef Name;

    if (CI->capturesVariableArrayType())
      Name = "vla";
    else if (CI->capturesThis())
      Name = "this";
    else
      Name = CI->getCapturedVar()->getName();

    // If this is a capture by copy the element type has to be the pointer to
    // the data.
    if (CI->capturesVariableByCopy() || CI->capturesVariableArrayType())
      ElemTy = Ctx.getPointerType(ElemTy);

    ArgsAddresses.push_back(CGF.CreateMemTemp(ElemTy, Name + ".addr"));
  }

  // If this region is sharing loop bounds we need to create the local variables
  // to store the right addresses.
  DoOnSharedLoopBounds(D, [&CGF, &Ctx, &ArgsAddresses](const VarDecl *LB,
                                                       const VarDecl *UB) {
    ArgsAddresses.push_back(
        CGF.CreateMemTemp(Ctx.getPointerType(LB->getType()), "prev.lb.addr"));
    ArgsAddresses.push_back(
        CGF.CreateMemTemp(Ctx.getPointerType(UB->getType()), "prev.ub.addr"));
  });

  auto &&L0ParallelGen = [this, &D, &DSI, &CS, &RD, &ArgsAddresses](
                             CodeGenFunction &CGF, PrePostActionTy &) {
    auto &Bld = CGF.Builder;

    // In the Level 0 regions, we need to get the record of the master thread.
    auto *DataAddr = CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(
            OMPRTL_NVPTX__kmpc_get_data_sharing_environment_frame),
        {GetMasterThreadID(CGF),
         Bld.getInt16(isOMPRuntimeInitialized() ? 1 : 0)});
    auto *RTy = CGF.getTypes().ConvertTypeForMem(DSI.MasterRecordType);
    auto *CastedDataAddr =
        Bld.CreateBitOrPointerCast(DataAddr, RTy->getPointerTo());

    // For each capture obtain the pointer by calculating the right offset in
    // the host record.
    unsigned ArgsIdx = 0;
    auto CurField = RD->field_begin();
    auto FI =
        DSI.MasterRecordType->getAs<RecordType>()->getDecl()->field_begin();
    for (CapturedStmt::const_capture_iterator CI = CS.capture_begin(),
                                              CE = CS.capture_end();
         CI != CE; ++CI, ++CurField) {
      const VarDecl *VD;
      if (CurField->hasCapturedVLAType() != CI->capturesVariableArrayType())
        assert(false && "These values should match!");

      if (CI->capturesVariableArrayType()){
        // Get declaration
        auto VAT = CurField->getCapturedVLAType();
        VD = DSI.getVLADecl(VAT->getSizeExpr());
      } else {
        VD = CI->capturesThis() ? nullptr : CI->getCapturedVar();
      }
      CreateAddressStoreForVariable(CGF, VD, FI->getType(), DSI, CastedDataAddr,
                                    ArgsAddresses[ArgsIdx], CI->getLocation());
      ++FI;
      ++ArgsIdx;
    }

    // Get the addresses of the loop bounds if required.
    DoOnSharedLoopBounds(D, [&CGF, &DSI, &CastedDataAddr, &ArgsAddresses,
                             &ArgsIdx](const VarDecl *LB, const VarDecl *UB) {
      CreateAddressStoreForVariable(CGF, LB, LB->getType(), DSI, CastedDataAddr,
                                    ArgsAddresses[ArgsIdx++],
                                    LB->getLocation());
      CreateAddressStoreForVariable(CGF, UB, UB->getType(), DSI, CastedDataAddr,
                                    ArgsAddresses[ArgsIdx++],
                                    UB->getLocation());
    });
  };

  auto &&L1ParallelGen = [this, &D, &DSI, &CS, &RD, &ArgsAddresses,
                          SourceLaneID](CodeGenFunction &CGF,
                                        PrePostActionTy &) {
    auto &Bld = CGF.Builder;

    // In the Level 1 regions, we need to get the record of the current worker
    // thread.
    auto *DataAddr = CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(
            OMPRTL_NVPTX__kmpc_get_data_sharing_environment_frame),
        {GetNVPTXThreadID(CGF),
         /*isOMPRuntimeInitialized=*/Bld.getInt16(1)});
    auto *RTy = CGF.getTypes().ConvertTypeForMem(DSI.WorkerWarpRecordType);
    auto *CastedDataAddr =
        Bld.CreateBitOrPointerCast(DataAddr, RTy->getPointerTo());

    // For each capture obtain the pointer by calculating the right offset in
    // the host record.
    unsigned ArgsIdx = 0;
    auto CurField = RD->field_begin();
    auto FI =
        DSI.MasterRecordType->getAs<RecordType>()->getDecl()->field_begin();
    for (CapturedStmt::const_capture_iterator CI = CS.capture_begin(),
                                              CE = CS.capture_end();
         CI != CE; ++CI, ++ArgsIdx, ++CurField) {
      const VarDecl *VD;
      if (CI->capturesVariableArrayType()){
        // Get declaration
        auto VAT = CurField->getCapturedVLAType();
        VD = DSI.getVLADecl(VAT->getSizeExpr());
      } else
        VD = CI->capturesThis() ? nullptr : CI->getCapturedVar();
      CreateAddressStoreForVariable(CGF, VD, FI->getType(), DSI, CastedDataAddr,
                                    ArgsAddresses[ArgsIdx], CI->getLocation(),
                                    SourceLaneID);
      FI++;
    }

    // Get the addresses of the loop bounds if required.
    DoOnSharedLoopBounds(D, [&CGF, &DSI, &CastedDataAddr, &ArgsAddresses,
                             &ArgsIdx, SourceLaneID](const VarDecl *LB,
                                                     const VarDecl *UB) {
      CreateAddressStoreForVariable(CGF, LB, LB->getType(), DSI, CastedDataAddr,
                                    ArgsAddresses[ArgsIdx++], LB->getLocation(),
                                    SourceLaneID);
      CreateAddressStoreForVariable(CGF, UB, UB->getType(), DSI, CastedDataAddr,
                                    ArgsAddresses[ArgsIdx++], UB->getLocation(),
                                    SourceLaneID);
    });
  };
  auto &&Sequential = [](CodeGenFunction &CGF, PrePostActionTy &) {
    // A sequential region does not use the wrapper.
  };

  auto &&ParallelLevelGen = [&CGF, &ParallelLevelAddr, &Int16QTy,
                             &D]() -> llvm::Value * {
    return CGF.EmitLoadOfScalar(ParallelLevelAddr, /*Volatile=*/false, Int16QTy,
                                D.getLocStart());
  };
  // In Simd we only support L1 level.
  if (IsSimd)
    emitParallelismLevelCode(CGF, ParallelLevelGen, Sequential, L1ParallelGen,
                             Sequential);
  else
    emitParallelismLevelCode(CGF, ParallelLevelGen, L0ParallelGen,
                             L1ParallelGen, Sequential);

  // Get the array of arguments.
  SmallVector<llvm::Value *, 8> Args;

  if (IsSimd) {
    auto *LaneID =
        CGF.EmitLoadOfScalar(CGF.GetAddrOfLocalVar(&WrapperLaneArg),
                             /*Volatile=*/false, Int32PtrQTy, D.getLocStart());
    auto *NumLanes =
        CGF.EmitLoadOfScalar(CGF.GetAddrOfLocalVar(&WrapperNumLanesArg),
                             /*Volatile=*/false, Int32PtrQTy, D.getLocStart());
    Args.push_back(LaneID);
    Args.push_back(NumLanes);
  } else {
    Args.push_back(llvm::Constant::getNullValue(CGM.Int32Ty->getPointerTo()));
    Args.push_back(llvm::Constant::getNullValue(CGM.Int32Ty->getPointerTo()));
  }

  // Get the addresses of the loop bounds if required.
  DoOnSharedLoopBounds(D, [&CGF, &Ctx, &CS, &ArgsAddresses,
                           &Args](const VarDecl *LB, const VarDecl *UB) {
    QualType Ty = Ctx.getPointerType(Ctx.getUIntPtrType());
    unsigned Idx = 0;
    for (const VarDecl *L : {LB, UB}) {
      auto *Arg =
          CGF.EmitLoadOfScalar(ArgsAddresses[CS.capture_size() + Idx],
                               /*Volatile=*/false, Ty, L->getLocation());
      // Bounds are passed by value, so we need to load the data.
      auto LV = CGF.MakeNaturalAlignAddrLValue(Arg, L->getType());
      Arg = CGF.EmitLoadOfScalar(LV, L->getLocation());
      auto ArgCast =
          CGF.Builder.CreateIntCast(Arg, CGF.SizeTy, /* isSigned = */ false);
      Args.push_back(ArgCast);
      ++Idx;
    }
  });

  auto CapInfo = DSI.CapturesValues.begin();
  auto FI = DSI.MasterRecordType->getAs<RecordType>()->getDecl()->field_begin();
  auto CI = CS.capture_begin();
  auto CurrentField = RD->field_begin();
  for (unsigned i = 0 /*, ArgIdx = 0*/; i < CS.capture_size();
       ++i, ++CI, ++CapInfo, ++CurrentField) {
    auto *Arg = CGF.EmitLoadOfScalar(ArgsAddresses[i], /*Volatile=*/false,
                                     Ctx.getPointerType(FI->getType()),
                                     CI->getLocation());

    const VarDecl *CapturedVar;
    if (CI->capturesVariableArrayType()) {
      auto VAT = CurrentField->getCapturedVLAType();
      CapturedVar = DSI.getVLADecl(VAT->getSizeExpr());
      auto CapturedTy = FI->getType();
      auto LV = CGF.MakeNaturalAlignAddrLValue(Arg, CapturedTy);
      Arg = CGF.EmitLoadOfScalar(LV, CI->getLocation());
    } else if (CI->capturesVariableByCopy()) {
      CapturedVar = CI->getCapturedVar();
    }

    // If this is a capture by value, we need to load the data. Additionally, if
    // its not a pointer we may need to cast it to uintptr.
    if (CI->capturesVariableByCopy()) {
      auto CapturedTy = FI->getType();
      auto LV = CGF.MakeNaturalAlignAddrLValue(Arg, CapturedTy);

      // If this is a value captured by reference in the outermost scope, we
      // have to load the address first.
      assert(CapInfo->first == CapturedVar &&
             "Using info of wrong declaration.");
      if (CapInfo->second == DataSharingInfo::DST_Ref)
        CapturedTy = CapturedVar->getType();

      auto CastLV = castValueToUintptr(CGF, CapturedTy, CapturedVar->getName(),
                                       LV, CI->getLocation());

      Arg = CGF.EmitLoadOfScalar(CastLV, CI->getLocation());
    }

    Args.push_back(Arg);

    ++FI;
  }

  emitOutlinedFunctionCall(CGF, D.getLocStart(), &OutlinedParallelFn, Args);
  CGF.FinishFunction();
  return Fn;
}

namespace {
static bool IsVolta(CodeGenModule &CGM) {
  StringRef TargetCPU = CGM.getTarget().getTargetOpts().CPU;
  return StringToCudaArch(TargetCPU) == CudaArch::SM_70;
}
}

// \brief Emit the code that each thread requires to execute when it encounters
// one of the three possible parallelism level. This also emits the required
// data sharing code for each level.
void CGOpenMPRuntimeNVPTX::emitParallelismLevelCode(
    CodeGenFunction &CGF,
    const llvm::function_ref<llvm::Value *()> &ParallelLevelGen,
    const RegionCodeGenTy &Level0, const RegionCodeGenTy &Level1,
    const RegionCodeGenTy &Sequential) {
  auto &Bld = CGF.Builder;

  // Flags that prevent code to be emitted if it can be proven that threads
  // cannot reach this function at a given level.
  //
  // FIXME: This current relies on a simple analysis that may not be correct if
  // we have function in a target region.
  bool OnlyInL0 = InL0();
  bool OnlyInL1 = InL1();
  bool OnlySequential = !IsOrphaned && !InL0() && !InL1();

  // Emit runtime checks if we cannot prove this code is reached only at a
  // certain parallelism level.
  //
  // For each level i the code will look like:
  //
  //   isLevel = icmp Level, i;
  //   br isLevel, .leveli.parallel, .next.parallel
  //
  // .leveli.parallel:
  //   ; code for level i + shared data code
  //   br .after.parallel
  //
  // .next.parallel

  bool CGForVolta = IsVolta(CGF.CGM);
  llvm::BasicBlock *AfterBB = CGF.createBasicBlock(".after.parallel");

  // Do we need to emit L0 code?
  if (!OnlyInL1 && !OnlySequential) {
    llvm::BasicBlock *LBB = CGF.createBasicBlock(".level0.parallel");
    llvm::BasicBlock *NextBB = nullptr;

    // Do we need runtime checks
    if (!OnlyInL0) {
      NextBB = CGF.createBasicBlock(".next.parallel");
      llvm::Value *ParallelLevel = ParallelLevelGen();
      auto *Cond = Bld.CreateICmpEQ(ParallelLevel, Bld.getInt16(0));
      Bld.CreateCondBr(Cond, LBB, NextBB);
    }

    CGF.EmitBlock(LBB);

    Level0(CGF);

    CGF.EmitBranch(AfterBB);
    if (NextBB)
      CGF.EmitBlock(NextBB);
  }

  bool SkipSequentialDetect = false;
  // Do we need to emit L1 code?
  if (!OnlyInL0 && !OnlySequential) {
    llvm::BasicBlock *LBB = CGF.createBasicBlock(".level1.parallel");
    llvm::BasicBlock *NextBB = nullptr;

    // Do we need runtime checks
    // Volta always requires a runtime check for a full warp.
    if (CGForVolta || !OnlyInL1) {
      NextBB = CGF.createBasicBlock(".next.parallel");
      llvm::Value *ParallelLevel = ParallelLevelGen();
      auto *Cond = Bld.CreateICmpEQ(ParallelLevel, Bld.getInt16(1));
      Bld.CreateCondBr(Cond, LBB, NextBB);
    }
    // On Volta, if generating L1 code, we also need the serialized parallel
    // code.  We must also prevent any further runtime calls to
    // getParallelLevel.
    SkipSequentialDetect = CGForVolta;

    CGF.EmitBlock(LBB);

    Level1(CGF);

    CGF.EmitBranch(AfterBB);
    if (NextBB)
      CGF.EmitBlock(NextBB);
  }

  // Do we need to emit sequential code?
  if (!OnlyInL0 && (CGForVolta || !OnlyInL1)) {
    llvm::BasicBlock *SeqBB = CGF.createBasicBlock(".sequential.parallel");

    // Do we need runtime checks
    if (!OnlySequential && !SkipSequentialDetect) {
      llvm::Value *ParallelLevel = ParallelLevelGen();
      auto *Cond = Bld.CreateICmpSGT(ParallelLevel, Bld.getInt16(1));
      Bld.CreateCondBr(Cond, SeqBB, AfterBB);
    }

    CGF.EmitBlock(SeqBB);
    Sequential(CGF);
  }

  CGF.EmitBlock(AfterBB);
}

void CGOpenMPRuntimeNVPTX::emitGenericParallelCall(
    CodeGenFunction &CGF, SourceLocation Loc, llvm::Value *OutlinedFn,
    ArrayRef<llvm::Value *> CapturedVars, const Expr *IfCond) {

  llvm::Function *Fn = cast<llvm::Function>(OutlinedFn);
  llvm::Function *WFn = WrapperFunctionsMap[Fn];
  assert(WFn && "Wrapper function does not exist??");

  // Force inline this outlined function at its call site.
  Fn->removeFnAttr(llvm::Attribute::NoInline);
  Fn->addFnAttr(llvm::Attribute::AlwaysInline);
  Fn->setLinkage(llvm::GlobalValue::InternalLinkage);

  // Emit code that does the data sharing changes in the beginning of the
  // function.
  createDataSharingPerFunctionInfrastructure(CGF);

  auto *RTLoc = emitUpdateLocation(CGF, Loc);
  auto &&L0ParallelGen = [this, WFn](CodeGenFunction &CGF, PrePostActionTy &) {
    CGBuilderTy &Bld = CGF.Builder;

    auto ID = Bld.CreateBitOrPointerCast(WFn, CGM.Int8PtrTy);

    // Prepare for parallel region. Indicate the outlined function.
    llvm::Value *IsOMPRuntimeInitialized =
        Bld.getInt16(isOMPRuntimeInitialized() ? 1 : 0);
    llvm::Value *Args[] = {ID, IsOMPRuntimeInitialized};
    CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_prepare_parallel),
        Args);

    // Activate workers.
    SyncCTAThreads(CGF);

    // Barrier at end of parallel region.
    SyncCTAThreads(CGF);

    // Remember for post-processing in worker loop.
    Work.push_back(WFn);
  };
  auto &&L1ParallelGen = [this, WFn, Loc](CodeGenFunction &CGF,
                                          PrePostActionTy &) {
    CGBuilderTy &Bld = CGF.Builder;
    clang::ASTContext &Ctx = CGF.getContext();

    Address IsFinal =
        CGF.CreateTempAlloca(CGF.Int8Ty, CharUnits::fromQuantity(1),
                             /*Name*/ "is_final");
    Address WorkSource =
        CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4),
                             /*Name*/ "work_source");
    Address ConvergentMask =
        CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4),
                             /*Name*/ "convergent_mask");
    llvm::APInt TaskBufferSize(/*numBits=*/32, TASK_STATE_SIZE);
    auto TaskBufferTy = Ctx.getConstantArrayType(
        Ctx.CharTy, TaskBufferSize, ArrayType::Normal, /*IndexTypeQuals=*/0);
    auto TaskState = CGF.CreateMemTemp(TaskBufferTy, CharUnits::fromQuantity(8),
                                       /*Name=*/"task_state")
                         .getPointer();
    CGF.InitTempAlloca(IsFinal, Bld.getInt8(/*C=*/0));
    CGF.InitTempAlloca(WorkSource, Bld.getInt32(/*C=*/-1));

    llvm::BasicBlock *DoBodyBB = CGF.createBasicBlock(".do.body");
    llvm::BasicBlock *ExecuteBB = CGF.createBasicBlock(".do.body.execute");
    llvm::BasicBlock *DoCondBB = CGF.createBasicBlock(".do.cond");
    llvm::BasicBlock *DoEndBB = CGF.createBasicBlock(".do.end");

    // Initialize WorkSource before each call to the parallel region.
    Bld.CreateStore(Bld.getInt32(/*C=*/-1), WorkSource);
    Bld.CreateStore(getNVPTXWarpActiveThreadsMask(CGF), ConvergentMask);

    CGF.EmitBranch(DoBodyBB);
    CGF.EmitBlock(DoBodyBB);
    auto ArrayDecay = Bld.CreateConstInBoundsGEP2_32(
        llvm::ArrayType::get(CGM.Int8Ty, TASK_STATE_SIZE), TaskState,
        /*Idx0=*/0, /*Idx1=*/0);
    auto *MaskVal = CGF.EmitLoadOfScalar(
        ConvergentMask, /*Volatile=*/false,
        Ctx.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/false), Loc);
    llvm::Value *Args[] = {ArrayDecay, MaskVal, IsFinal.getPointer(),
                           WorkSource.getPointer()};
    llvm::Value *IsActive =
        CGF.EmitRuntimeCall(createNVPTXRuntimeFunction(
                                OMPRTL_NVPTX__kmpc_kernel_convergent_parallel),
                            Args);
    Bld.CreateCondBr(IsActive, ExecuteBB, DoCondBB);

    CGF.EmitBlock(ExecuteBB);

    // Execute the work, and pass the thread source from where the data should
    // be used.
    auto *SourceThread = CGF.EmitLoadOfScalar(
        WorkSource, /*Volatile=*/false,
        Ctx.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/false), Loc);
    emitOutlinedFunctionCall(CGF, Loc, WFn,
                             {Bld.getInt16(/*ParallelLevel=*/1), SourceThread});
    ArrayDecay = Bld.CreateConstInBoundsGEP2_32(
        llvm::ArrayType::get(CGM.Int8Ty, TASK_STATE_SIZE), TaskState,
        /*Idx0=*/0, /*Idx1=*/0);
    llvm::Value *EndArgs[] = {ArrayDecay};
    CGF.EmitRuntimeCall(createNVPTXRuntimeFunction(
                            OMPRTL_NVPTX__kmpc_kernel_end_convergent_parallel),
                        EndArgs);
    CGF.EmitBranch(DoCondBB);

    CGF.EmitBlock(DoCondBB);
    llvm::Value *IsDone = Bld.CreateICmpEQ(Bld.CreateLoad(IsFinal),
                                           Bld.getInt8(/*C=*/1), "is_done");
    Bld.CreateCondBr(IsDone, DoEndBB, DoBodyBB);

    CGF.EmitBlock(DoEndBB);
  };

  auto &&SeqGen = [this, Fn, &CapturedVars, &RTLoc, &Loc](CodeGenFunction &CGF,
                                                          PrePostActionTy &) {
    auto DL = CGM.getDataLayout();
    auto ThreadID = getThreadID(CGF, Loc);
    // Build calls:
    // __kmpc_serialized_parallel(&Loc, GTid);
    llvm::Value *Args[] = {RTLoc, ThreadID};
    CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_serialized_parallel),
        Args);

    llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;
    OutlinedFnArgs.push_back(
        llvm::Constant::getNullValue(CGM.Int32Ty->getPointerTo()));
    OutlinedFnArgs.push_back(
        llvm::Constant::getNullValue(CGM.Int32Ty->getPointerTo()));
    OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());
    emitOutlinedFunctionCall(CGF, Loc, Fn, OutlinedFnArgs);

    // __kmpc_end_serialized_parallel(&Loc, GTid);
    llvm::Value *EndArgs[] = {emitUpdateLocation(CGF, Loc), ThreadID};
    CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_end_serialized_parallel),
        EndArgs);
  };

  auto &&ThenGen = [this, &Loc, &L0ParallelGen, &L1ParallelGen,
                    &SeqGen](CodeGenFunction &CGF, PrePostActionTy &) {
    auto &&ParallelLevelGen = [this, &CGF, &Loc]() -> llvm::Value * {
      return getParallelLevel(CGF, Loc);
    };
    emitParallelismLevelCode(CGF, ParallelLevelGen, L0ParallelGen,
                             L1ParallelGen, SeqGen);
  };

  if (IfCond) {
    emitOMPIfClause(CGF, IfCond, ThenGen, SeqGen);
  } else {
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    RegionCodeGenTy ThenRCG(ThenGen);
    ThenRCG(CGF);
  }
}

void CGOpenMPRuntimeNVPTX::emitSPMDParallelCall(
    CodeGenFunction &CGF, SourceLocation Loc, llvm::Value *OutlinedFn,
    ArrayRef<llvm::Value *> CapturedVars, const Expr *IfCond) {
  if (InL0()) {
    // Just call the outlined function to execute the parallel region.
    //
    // Note that the target region is started with 1 thread if executing
    // in a serialized parallel region, so the IfCond can be ignored.

    // OutlinedFn(&GTid, &zero, CapturedStruct);
    auto ThreadIDAddr = emitThreadIDAddress(CGF, Loc);
    Address ZeroAddr =
        CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4),
                             /*Name*/ ".zero.addr");
    CGF.InitTempAlloca(ZeroAddr, CGF.Builder.getInt32(/*C*/ 0));
    llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;
    OutlinedFnArgs.push_back(ThreadIDAddr.getPointer());
    OutlinedFnArgs.push_back(ZeroAddr.getPointer());
    OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());
    emitOutlinedFunctionCall(CGF, Loc, OutlinedFn, OutlinedFnArgs);
  } else {
    emitGenericParallelCall(CGF, Loc, OutlinedFn, CapturedVars, IfCond);
  }
}

void CGOpenMPRuntimeNVPTX::emitParallelCall(
    CodeGenFunction &CGF, SourceLocation Loc, llvm::Value *OutlinedFn,
    ArrayRef<llvm::Value *> CapturedVars, const Expr *IfCond) {
  if (!CGF.HaveInsertPoint())
    return;

  if (isSPMDExecutionMode())
    emitSPMDParallelCall(CGF, Loc, OutlinedFn, CapturedVars, IfCond);
  else
    emitGenericParallelCall(CGF, Loc, OutlinedFn, CapturedVars, IfCond);
}

void CGOpenMPRuntimeNVPTX::emitSimdCall(CodeGenFunction &CGF,
                                        SourceLocation Loc,
                                        llvm::Value *OutlinedFn,
                                        ArrayRef<llvm::Value *> CapturedVars) {
  if (!CGF.HaveInsertPoint())
    return;

  llvm::Function *Fn = cast<llvm::Function>(OutlinedFn);
  llvm::Function *WFn = WrapperFunctionsMap[Fn];
  assert(WFn && "Wrapper function does not exist??");

  // Force inline this outlined function at its call site.
  Fn->removeFnAttr(llvm::Attribute::NoInline);
  Fn->addFnAttr(llvm::Attribute::AlwaysInline);
  Fn->setLinkage(llvm::GlobalValue::InternalLinkage);

  // Emit code that does the data sharing changes in the beginning of the
  // function.
  createDataSharingPerFunctionInfrastructure(CGF);

  auto &&L1SimdGen = [this, WFn, Loc](CodeGenFunction &CGF, PrePostActionTy &) {
    CGBuilderTy &Bld = CGF.Builder;
    clang::ASTContext &Ctx = CGF.getContext();

    Address IsFinal =
        CGF.CreateTempAlloca(CGF.Int8Ty, CharUnits::fromQuantity(1),
                             /*Name*/ "is_final");
    Address WorkSource =
        CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4),
                             /*Name*/ "work_source");
    Address LaneId =
        CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4),
                             /*Name*/ "lane_id");
    Address NumLanes =
        CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4),
                             /*Name*/ "num_lanes");
    llvm::APInt TaskBufferSize(/*numBits=*/32, SIMD_STATE_SIZE);
    auto TaskBufferTy = Ctx.getConstantArrayType(
        Ctx.CharTy, TaskBufferSize, ArrayType::Normal, /*IndexTypeQuals=*/0);
    auto TaskState = CGF.CreateMemTemp(TaskBufferTy, CharUnits::fromQuantity(8),
                                       /*Name=*/"task_state")
                         .getPointer();
    auto ConvergentMask =
        CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4),
                             /*Name*/ "convergent_mask");
    CGF.InitTempAlloca(IsFinal, Bld.getInt8(/*C=*/0));
    CGF.InitTempAlloca(WorkSource, Bld.getInt32(/*C=*/-1));

    llvm::BasicBlock *DoBodyBB = CGF.createBasicBlock(".do.body");
    llvm::BasicBlock *ExecuteBB = CGF.createBasicBlock(".do.body.execute");
    llvm::BasicBlock *DoCondBB = CGF.createBasicBlock(".do.cond");
    llvm::BasicBlock *DoEndBB = CGF.createBasicBlock(".do.end");

    // Initialize WorkSource before each call to the simd region.
    Bld.CreateStore(Bld.getInt32(/*C=*/-1), WorkSource);
    Bld.CreateStore(getNVPTXWarpActiveThreadsMask(CGF), ConvergentMask);

    CGF.EmitBranch(DoBodyBB);
    CGF.EmitBlock(DoBodyBB);
    auto ArrayDecay = Bld.CreateConstInBoundsGEP2_32(
        llvm::ArrayType::get(CGM.Int8Ty, SIMD_STATE_SIZE), TaskState,
        /*Idx0=*/0, /*Idx1=*/0);
    auto *MaskVal = CGF.EmitLoadOfScalar(
        ConvergentMask, /*Volatile=*/false,
        Ctx.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/false), Loc);
    llvm::Value *Args[] = {ArrayDecay,           MaskVal,
                           IsFinal.getPointer(), WorkSource.getPointer(),
                           LaneId.getPointer(),  NumLanes.getPointer()};
    llvm::Value *IsActive = CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_convergent_simd),
        Args);
    Bld.CreateCondBr(IsActive, ExecuteBB, DoCondBB);

    CGF.EmitBlock(ExecuteBB);

    llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;

    // We are in an L1 parallel region.
    auto *ParallelLevel = Bld.getInt16(1);
    auto *SourceThread = CGF.EmitLoadOfScalar(
        WorkSource, /*Volatile=*/false,
        Ctx.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/false), Loc);
    OutlinedFnArgs.push_back(ParallelLevel);
    OutlinedFnArgs.push_back(SourceThread);
    OutlinedFnArgs.push_back(LaneId.getPointer());
    OutlinedFnArgs.push_back(NumLanes.getPointer());
    emitOutlinedFunctionCall(CGF, Loc, WFn, OutlinedFnArgs);
    ArrayDecay = Bld.CreateConstInBoundsGEP2_32(
        llvm::ArrayType::get(CGM.Int8Ty, SIMD_STATE_SIZE), TaskState,
        /*Idx0=*/0, /*Idx1=*/0);
    llvm::Value *EndArgs[] = {ArrayDecay};
    CGF.EmitRuntimeCall(createNVPTXRuntimeFunction(
                            OMPRTL_NVPTX__kmpc_kernel_end_convergent_simd),
                        EndArgs);
    CGF.EmitBranch(DoCondBB);

    CGF.EmitBlock(DoCondBB);
    llvm::Value *IsDone = Bld.CreateICmpEQ(Bld.CreateLoad(IsFinal),
                                           Bld.getInt8(/*C=*/1), "is_done");
    Bld.CreateCondBr(IsDone, DoEndBB, DoBodyBB);

    CGF.EmitBlock(DoEndBB);
  };

  auto &&SeqGen = [Fn, &CapturedVars, Loc](CodeGenFunction &CGF,
                                           PrePostActionTy &) {
    CGBuilderTy &Bld = CGF.Builder;
    Address LaneId =
        CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4),
                             /*Name*/ "lane_id");
    Address NumLanes =
        CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4),
                             /*Name*/ "num_lanes");

    CGF.InitTempAlloca(LaneId, Bld.getInt32(/*C=*/0));
    CGF.InitTempAlloca(NumLanes, Bld.getInt32(/*C=*/1));

    llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;
    OutlinedFnArgs.push_back(LaneId.getPointer());
    OutlinedFnArgs.push_back(NumLanes.getPointer());
    OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());
    CGF.CGM.getOpenMPRuntime().emitOutlinedFunctionCall(CGF, Loc, Fn,
                                                        OutlinedFnArgs);
  };

  CodeGenFunction::RunCleanupsScope Scope(CGF);
  auto &&ParallelLevelGen = [this, &CGF, &Loc]() -> llvm::Value * {
    return getParallelLevel(CGF, Loc);
  };
  emitParallelismLevelCode(CGF, ParallelLevelGen, SeqGen, L1SimdGen, SeqGen);
}

/// \brief Emits a critical region.
/// \param CriticalName Name of the critical region.
/// \param CriticalOpGen Generator for the statement associated with the given
/// critical region.
/// \param Hint Value of the 'hint' clause (optional).
void CGOpenMPRuntimeNVPTX::emitCriticalRegion(
    CodeGenFunction &CGF, StringRef CriticalName,
    const RegionCodeGenTy &CriticalOpGen, SourceLocation Loc,
    const Expr *Hint) {

  auto *LoopBB = CGF.createBasicBlock("omp.critical.loop");
  auto *TestBB = CGF.createBasicBlock("omp.critical.test");
  auto *SyncBB = CGF.createBasicBlock("omp.critical.sync");
  auto *BodyBB = CGF.createBasicBlock("omp.critical.body");
  auto *ExitBB = CGF.createBasicBlock("omp.critical.exit");

  /// Fetch team-local id of the thread.
  auto ThreadID = GetNVPTXThreadID(CGF);

  /// Get the width of the team.
  auto TeamWidth = GetNVPTXNumThreads(CGF);

  /// Initialise the counter variable for the loop.
  auto Int32Ty =
      CGF.getContext().getIntTypeForBitwidth(/*DestWidth*/ 32, /*Signed*/ true);
  auto Counter = CGF.CreateMemTemp(Int32Ty, "critical_counter");
  auto CounterLVal =
      CGF.MakeNaturalAlignAddrLValue(Counter.getPointer(), Int32Ty);
  CGF.EmitStoreOfScalar(llvm::ConstantInt::get(CGM.Int32Ty, 0), CounterLVal);
  CGF.EmitBranch(LoopBB);

  /// Block checks if loop counter exceeds upper bound.
  CGF.EmitBlock(LoopBB);
  auto *CounterVal = CGF.EmitLoadOfScalar(CounterLVal, Loc);
  auto *CmpLoopBound = CGF.Builder.CreateICmpSLT(CounterVal, TeamWidth);
  CGF.Builder.CreateCondBr(CmpLoopBound, TestBB, ExitBB);

  /// Block tests which single thread should execute region, and
  /// which threads should go straight to synchronisation point.
  CGF.EmitBlock(TestBB);
  CounterVal = CGF.EmitLoadOfScalar(CounterLVal, Loc);
  auto *CmpThreadToCounter = CGF.Builder.CreateICmpEQ(ThreadID, CounterVal);
  CGF.Builder.CreateCondBr(CmpThreadToCounter, BodyBB, SyncBB);

  /// Block emits the body of the critical region.
  CGF.EmitBlock(BodyBB);

  /// Output the critical statement.
  CriticalOpGen(CGF);

  /// After the body surrounded by the critical region, the single executing
  /// thread will jump to the synchronisation point.
  CGF.EmitBranch(SyncBB);

  /// Block waits for all threads in current team to finish then increments
  /// the counter variable and returns to the loop.
  CGF.EmitBlock(SyncBB);
  GetNVPTXCTABarrier(CGF);

  auto *IncCounterVal =
      CGF.Builder.CreateNSWAdd(CounterVal, CGF.Builder.getInt32(1));
  CGF.EmitStoreOfScalar(IncCounterVal, CounterLVal);
  CGF.EmitBranch(LoopBB);

  /// Block that is reached when  all threads in the team complete the region.
  CGF.EmitBlock(ExitBB, /*IsFinished=*/true);
}

namespace {
template <typename T>
static T selectRuntimeCall(bool IsSPMDExecutionMode,
                           bool IsOMPRuntimeInitialized, ArrayRef<T> V) {
  if (IsOMPRuntimeInitialized) {
    assert(V.size() >= 3 && "Invalid argument to select runtime call.");
    return V[2];
  } else /*Elided runtime*/ {
    assert(V.size() >= 2 && "Invalid argument to select runtime call.");
    return IsSPMDExecutionMode ? V[0] : V[1];
  }
}
}

llvm::Constant *
CGOpenMPRuntimeNVPTX::createForStaticInitFunction(unsigned IVSize,
                                                  bool IVSigned) {
  assert((IVSize == 32 || IVSize == 64) &&
         "IV size is not compatible with the omp runtime");
  auto Name =
      IVSize == 32
          ? (IVSigned ? selectRuntimeCall<StringRef>(
                            isSPMDExecutionMode(), isOMPRuntimeInitialized(),
                            {"__kmpc_for_static_init_4_simple_spmd",
                             "__kmpc_for_static_init_4_simple_generic",
                             "__kmpc_for_static_init_4"})
                      : selectRuntimeCall<StringRef>(
                            isSPMDExecutionMode(), isOMPRuntimeInitialized(),
                            {"__kmpc_for_static_init_4u_simple_spmd",
                             "__kmpc_for_static_init_4u_simple_generic",
                             "__kmpc_for_static_init_4u"}))
          : (IVSigned ? selectRuntimeCall<StringRef>(
                            isSPMDExecutionMode(), isOMPRuntimeInitialized(),
                            {"__kmpc_for_static_init_8_simple_spmd",
                             "__kmpc_for_static_init_8_simple_generic",
                             "__kmpc_for_static_init_8"})
                      : selectRuntimeCall<StringRef>(
                            isSPMDExecutionMode(), isOMPRuntimeInitialized(),
                            {"__kmpc_for_static_init_8u_simple_spmd",
                             "__kmpc_for_static_init_8u_simple_generic",
                             "__kmpc_for_static_init_8u"}));
  auto ITy = IVSize == 32 ? CGM.Int32Ty : CGM.Int64Ty;
  auto PtrTy = llvm::PointerType::getUnqual(ITy);
  llvm::Type *TypeParams[] = {
      getIdentTyPointerTy(),                     // loc
      CGM.Int32Ty,                               // tid
      CGM.Int32Ty,                               // schedtype
      llvm::PointerType::getUnqual(CGM.Int32Ty), // p_lastiter
      PtrTy,                                     // p_lower
      PtrTy,                                     // p_upper
      PtrTy,                                     // p_stride
      ITy,                                       // incr
      ITy                                        // chunk
  };
  llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
  return CGM.CreateRuntimeFunction(FnTy, Name);
}

//
// Generate optimized code resembling static schedule with chunk size of 1
// whenever the standard gives us freedom.  This allows maximum coalescing on
// the NVPTX target.
//
bool CGOpenMPRuntimeNVPTX::generateCoalescedSchedule(
    OpenMPScheduleClauseKind ScheduleKind, bool ChunkSizeOne,
    bool Ordered) const {
  return !Ordered && (ScheduleKind == OMPC_SCHEDULE_unknown ||
                      ScheduleKind == OMPC_SCHEDULE_auto ||
                      (ScheduleKind == OMPC_SCHEDULE_static && ChunkSizeOne));
}

//
// Generate optimized code resembling dist_schedule(static, num_threads) and
// schedule(static, 1) whenever the standard gives us freedom.  This allows
// maximum coalescing on the NVPTX target and minimum loop overhead.
//
// Only possible in SPMD mode.
//
bool CGOpenMPRuntimeNVPTX::generateCoalescedSchedule(
    OpenMPDistScheduleClauseKind DistScheduleKind,
    OpenMPScheduleClauseKind ScheduleKind, bool DistChunked, bool ChunkSizeOne,
    bool Ordered) const {
  return isSPMDExecutionMode() &&
         DistScheduleKind == OMPC_DIST_SCHEDULE_unknown &&
         generateCoalescedSchedule(ScheduleKind, ChunkSizeOne, Ordered);
}

bool CGOpenMPRuntimeNVPTX::requiresBarrier(const OMPLoopDirective &S) const {
  const bool Ordered = S.getSingleClause<OMPOrderedClause>() != nullptr;
  OpenMPScheduleClauseKind ScheduleKind = OMPC_SCHEDULE_unknown;
  if (auto *C = S.getSingleClause<OMPScheduleClause>())
    ScheduleKind = C->getScheduleKind();
  return Ordered || ScheduleKind == OMPC_SCHEDULE_dynamic ||
         ScheduleKind == OMPC_SCHEDULE_guided;
}

/// \brief Values for bit flags used in the ident_t to describe the fields.
/// All enumeric elements are named and described in accordance with the code
/// from http://llvm.org/svn/llvm-project/openmp/trunk/runtime/src/kmp.h
enum OpenMPLocationFlags {
  /// \brief Implicit barrier in 'for' directive.
  OMP_IDENT_BARRIER_IMPL_FOR = 0x40,
  /// \brief Implicit barrier in code.
  OMP_IDENT_BARRIER_IMPL = 0x40,
};

// Handle simple barrier case when runtime is not available.
void CGOpenMPRuntimeNVPTX::emitBarrierCall(CodeGenFunction &CGF,
                                           SourceLocation Loc,
                                           OpenMPDirectiveKind Kind,
                                           bool EmitChecks,
                                           bool ForceSimpleCall) {
  if (!CGF.HaveInsertPoint())
    return;

  bool IsSimpleBarrier =
      Kind == OMPD_for || Kind == OMPD_barrier || Kind == OMPD_unknown;
  if (auto *OMPRegionInfo =
          dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo))
    if (!ForceSimpleCall && OMPRegionInfo->hasCancel())
      IsSimpleBarrier = false;

  if (!isOMPRuntimeInitialized() && IsSimpleBarrier) {
    // Build call __kmpc_barrier_simple(loc, thread_id);
    unsigned Flags;
    if (Kind == OMPD_for)
      Flags = OMP_IDENT_BARRIER_IMPL_FOR;
    else
      Flags = OMP_IDENT_BARRIER_IMPL;
    llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc, Flags),
                           getThreadID(CGF, Loc)};
    auto Name = selectRuntimeCall<OpenMPRTLFunctionNVPTX>(
        isSPMDExecutionMode(), /*isOMPRuntimeInitialized=*/false,
        {OMPRTL_NVPTX__kmpc_barrier_simple_spmd,
         OMPRTL_NVPTX__kmpc_barrier_simple_generic});
    CGF.EmitRuntimeCall(createNVPTXRuntimeFunction(Name), Args);
  } else {
    return CGOpenMPRuntime::emitBarrierCall(CGF, Loc, Kind, EmitChecks,
                                            ForceSimpleCall);
  }
}

CGOpenMPRuntimeNVPTX::CGOpenMPRuntimeNVPTX(CodeGenModule &CGM)
    : CGOpenMPRuntime(CGM), IsOrphaned(true), ParallelNestingLevel(0),
      IsOMPRuntimeInitialized(true), CurrMode(ExecutionMode::UNKNOWN) {
  if (!CGM.getLangOpts().OpenMPIsDevice)
    llvm_unreachable("OpenMP NVPTX can only handle device code.");
}

void CGOpenMPRuntimeNVPTX::emitNumTeamsClause(CodeGenFunction &CGF,
                                              const Expr *NumTeams,
                                              const Expr *ThreadLimit,
                                              SourceLocation Loc) {}

void CGOpenMPRuntimeNVPTX::emitInitDSBlock(CodeGenFunction &CGF) {
  auto BB = CGF.createBasicBlock("omp.init.ds");
  CGF.EmitBranch(BB);
  CGF.EmitBlock(BB);

  // Save basic block because names are immediately discarded in Release
  // builds that pass in -discard-value-names.
  auto &DSInfo = DataSharingFunctionInfoMap[CGF.CurFn];
  assert(DSInfo.InitDSBlock == nullptr && "Didn't expect data sharing!");
  DSInfo.InitDSBlock = BB;
}

static void emitPostUpdateForReductionClause(
    CodeGenFunction &CGF, const OMPExecutableDirective &D,
    const llvm::function_ref<llvm::Value *(CodeGenFunction &)> &CondGen) {
  if (!CGF.HaveInsertPoint())
    return;
  llvm::BasicBlock *DoneBB = nullptr;
  for (const auto *C : D.getClausesOfKind<OMPReductionClause>()) {
    if (auto *PostUpdate = C->getPostUpdateExpr()) {
      if (!DoneBB) {
        if (auto *Cond = CondGen(CGF)) {
          // If the first post-update expression is found, emit conditional
          // block if it was requested.
          auto *ThenBB = CGF.createBasicBlock(".omp.reduction.pu");
          DoneBB = CGF.createBasicBlock(".omp.reduction.pu.done");
          CGF.Builder.CreateCondBr(Cond, ThenBB, DoneBB);
          CGF.EmitBlock(ThenBB);
        }
      }
      CGF.EmitIgnoredExpr(PostUpdate);
    }
  }
  if (DoneBB)
    CGF.EmitBlock(DoneBB, /*IsFinished=*/true);
}

void CGOpenMPRuntimeNVPTX::emitTeamsCall(CodeGenFunction &CGF,
                                         const OMPExecutableDirective &D,
                                         SourceLocation Loc,
                                         llvm::Value *OutlinedFn,
                                         ArrayRef<llvm::Value *> CapturedVars) {
  if (isSPMDExecutionMode()) {
    // OutlinedFn(&GTid, &zero, CapturedStruct);
    auto ThreadIDAddr = emitThreadIDAddress(CGF, Loc);
    Address ZeroAddr =
        CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4),
                             /*Name*/ ".zero.addr");
    CGF.InitTempAlloca(ZeroAddr, CGF.Builder.getInt32(/*C*/ 0));
    llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;
    OutlinedFnArgs.push_back(ThreadIDAddr.getPointer());
    OutlinedFnArgs.push_back(ZeroAddr.getPointer());
    OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());
    emitOutlinedFunctionCall(CGF, Loc, OutlinedFn, OutlinedFnArgs);
  } else if (D.getDirectiveKind() == OMPD_teams_distribute ||
             D.getDirectiveKind() == OMPD_target_teams_distribute) {
    // This code generation is a duplication of the one in CGStmtOpenMP.cpp
    // and it has to be removed once the sharing from teams distribute to
    // any contained worksharing loop works smoothly.
    auto &&CGDistributeInlined = [this, &D](CodeGenFunction &CGF, PrePostActionTy &) {
      CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
      CGF.EmitOMPReductionClauseInit(D, PrivateScope);
      emitInitDSBlock(CGF);
      (void)PrivateScope.Privatize();
      auto &&CGDistributeLoop = [&D](CodeGenFunction &CGF, PrePostActionTy &) {
        CGF.EmitOMPDistributeLoop(*(dyn_cast<OMPLoopDirective>(&D)),
                                  [](CodeGenFunction &, PrePostActionTy &) {});
      };
      CGF.CGM.getOpenMPRuntime().emitInlinedDirective(CGF, OMPD_distribute,
                                                      CGDistributeLoop,
                                                      /*HasCancel=*/false);
      CGF.EmitOMPReductionClauseFinal(D, D.getDirectiveKind());
    };
    emitInlinedDirective(CGF, D.getDirectiveKind(), CGDistributeInlined);
    emitPostUpdateForReductionClause(
        CGF, D, [](CodeGenFunction &) -> llvm::Value * { return nullptr; });
  } else if (D.getDirectiveKind() == OMPD_teams_distribute_simd ||
             D.getDirectiveKind() == OMPD_target_teams_distribute_simd) {
    // This code generation is a duplication of the one in CGStmtOpenMP.cpp
    // and it has to be removed once the sharing from teams distribute to
    // any contained worksharing loop works smoothly.
    auto &&CGDistributeInlined = [this, &D](CodeGenFunction &CGF, PrePostActionTy &) {
      CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
      (void)CGF.EmitOMPFirstprivateClause(D, PrivateScope);
      CGF.EmitOMPPrivateClause(D, PrivateScope);
      CGF.EmitOMPReductionClauseInit(D, PrivateScope);
      emitInitDSBlock(CGF);
      (void)PrivateScope.Privatize();
      auto &&CGDistributeLoop = [&D](CodeGenFunction &CGF, PrePostActionTy &) {
        auto &&CGSimd = [&D](CodeGenFunction &CGF, PrePostActionTy &) {
          CGF.EmitOMPSimdLoop(*(dyn_cast<OMPLoopDirective>(&D)), false);
        };
        CGF.EmitOMPDistributeLoop(*(dyn_cast<OMPLoopDirective>(&D)), CGSimd);
      };
      CGF.CGM.getOpenMPRuntime().emitInlinedDirective(CGF, OMPD_distribute,
                                                      CGDistributeLoop,
                                                      /*HasCancel=*/false);
      CGF.EmitOMPReductionClauseFinal(D, D.getDirectiveKind());
    };
    emitInlinedDirective(CGF, D.getDirectiveKind(), CGDistributeInlined);
    emitPostUpdateForReductionClause(
        CGF, D, [](CodeGenFunction &) -> llvm::Value * { return nullptr; });
  } else {
    // Just emit the statements in the teams region inlined.
    // This has to be removed too when data sharing is fixed.
    auto &&CodeGen = [this, &D](CodeGenFunction &CGF, PrePostActionTy &) {
      CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
      (void)CGF.EmitOMPFirstprivateClause(D, PrivateScope);
      CGF.EmitOMPPrivateClause(D, PrivateScope);
      CGF.EmitOMPReductionClauseInit(D, PrivateScope);
      emitInitDSBlock(CGF);
      (void)PrivateScope.Privatize();
      const auto *CS = cast<CapturedStmt>(D.getAssociatedStmt());
      if (D.hasClausesOfKind<OMPDependClause>() &&
          isOpenMPTargetExecutionDirective(D.getDirectiveKind()))
        CS = cast<CapturedStmt>(CS->getCapturedStmt());
      CGF.EmitStmt(CS->getCapturedStmt());
      CGF.EmitOMPReductionClauseFinal(D, OMPD_teams);
    };

    emitInlinedDirective(CGF, OMPD_teams, CodeGen);
    emitPostUpdateForReductionClause(
        CGF, D, [](CodeGenFunction &) -> llvm::Value * { return nullptr; });
  }
}

llvm::Function *CGOpenMPRuntimeNVPTX::emitRegistrationFunction() {
  auto &Ctx = CGM.getContext();
  unsigned PointerAlign = Ctx.getTypeAlignInChars(Ctx.VoidPtrTy).getQuantity();
  unsigned Int32Align =
      Ctx.getTypeAlignInChars(
             Ctx.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/true))
          .getQuantity();

  auto *SlotTy = getDataSharingSlotTy();

  // Scan all the functions that have data sharing info.
  for (auto &DS : DataSharingFunctionInfoMap) {
    llvm::Function *Fn = DS.first;
    DataSharingFunctionInfo &DSI = DS.second;

    llvm::BasicBlock &HeaderBB = Fn->front();

    llvm::Instruction *SharedDataInfrastructureInsertPoint = nullptr;
    bool hasOMPInitDSBlock = (DSI.InitDSBlock != nullptr);
    if (hasOMPInitDSBlock)
      SharedDataInfrastructureInsertPoint = &(*DSI.InitDSBlock->begin());

    // Find the last alloca and the last replacement that is not an alloca.
    llvm::Instruction *LastAlloca = nullptr;
    llvm::Instruction *LastNonAllocaReplacement = nullptr;
    llvm::Instruction *LastNonAllocaNonRefReplacement = nullptr;

    for (auto &I : HeaderBB) {
      if (isa<llvm::AllocaInst>(I)) {
        LastAlloca = &I;
        continue;
      }

      auto It = std::find_if(
          DSI.ValuesToBeReplaced.begin(), DSI.ValuesToBeReplaced.end(),
          [&I](std::pair<llvm::Value *, bool> &P) { return P.first == &I; });
      if (It == DSI.ValuesToBeReplaced.end())
        continue;

      LastNonAllocaReplacement = cast<llvm::Instruction>(It->first);

      if (!It->second)
        LastNonAllocaNonRefReplacement = LastNonAllocaReplacement;
    }

    llvm::Instruction *DSInsertPtr = nullptr;
    if (SharedDataInfrastructureInsertPoint)
      DSInsertPtr = SharedDataInfrastructureInsertPoint;

    // We will start inserting after the first alloca or at the beginning of the
    // function.
    llvm::Instruction *InsertPtr = nullptr;
    if (LastAlloca)
      InsertPtr = LastAlloca->getNextNode();
    else
      InsertPtr = &(*HeaderBB.begin());

    assert(InsertPtr && "Empty function???");

    // Helper to emit the initializaion code at the provided insertion point.
    auto &&InitializeEntryPoint = [this, &DSI](llvm::Instruction *&InsertPtr) {
      assert(DSI.EntryWorkerFunction &&
             "All entry function are expected to have an worker function.");
      assert(DSI.EntryExitBlock &&
             "All entry function are expected to have an exit basic block.");

      auto *ShouldReturnImmediatelly = llvm::CallInst::Create(
          createKernelInitializerFunction(DSI.EntryWorkerFunction,
                                          DSI.RequiresOMPRuntime),
          "", InsertPtr);
      auto *Cond = llvm::ICmpInst::Create(
          llvm::CmpInst::ICmp, llvm::CmpInst::ICMP_EQ, ShouldReturnImmediatelly,
          llvm::ConstantInt::get(CGM.Int32Ty, 1), "", InsertPtr);
      auto *CurrentBB = InsertPtr->getParent();
      auto *MasterBB = CurrentBB->splitBasicBlock(InsertPtr, ".master");

      // Adjust the terminator of the current block.
      CurrentBB->getTerminator()->eraseFromParent();
      llvm::BranchInst::Create(MasterBB, DSI.EntryExitBlock, Cond, CurrentBB);

      // Continue inserting in the master basic block.
      InsertPtr = &*MasterBB->begin();
    };

    // If there is nothing to share, and this is an entry point, we should
    // initialize the data sharing logic anyways.
    if (!DSI.InitializationFunction && DSI.IsEntryPoint) {
      if (DSInsertPtr)
        InitializeEntryPoint(DSInsertPtr);
      else
        InitializeEntryPoint(InsertPtr);
      continue;
    }

    SmallVector<llvm::Value *, 16> InitArgs;
    SmallVector<std::pair<llvm::Value *, llvm::Value *>, 16> Replacements;

    // Create the saved slot/stack/frame/active thread variables.
    InitArgs.push_back(
        new llvm::AllocaInst(SlotTy->getPointerTo(), /*ArraySize=*/nullptr,
                             PointerAlign, "data_share_slot_saved", InsertPtr));
    InitArgs.push_back(
        new llvm::AllocaInst(CGM.VoidPtrTy, /*ArraySize=*/nullptr, PointerAlign,
                             "data_share_stack_saved", InsertPtr));
    InitArgs.push_back(
        new llvm::AllocaInst(CGM.VoidPtrTy, /*ArraySize=*/nullptr, PointerAlign,
                             "data_share_frame_saved", InsertPtr));
    InitArgs.push_back(
        new llvm::AllocaInst(CGM.Int32Ty, /*ArraySize=*/nullptr, Int32Align,
                             "data_share_active_thd_saved", InsertPtr));

    // Create the remaining arguments. One if it is a reference sharing (the
    // reference itself), two otherwise (the address of the replacement and the
    // value to be replaced).
    for (auto &VR : DSI.ValuesToBeReplaced) {
      auto *Replacement = VR.first;
      bool IsReference = VR.second;
      // Is it a reference? If not, create the address alloca.
      if (!IsReference) {
        InitArgs.push_back(new llvm::AllocaInst(
            Replacement->getType(), /*ArraySize=*/nullptr, PointerAlign,
            Replacement->getName() + ".shared", InsertPtr));
        // We will have to replace the uses of Replacement by the load of new
        // alloca.
        Replacements.push_back(std::make_pair(Replacement, InitArgs.back()));
      }
      InitArgs.push_back(Replacement);
    }

    // Save the insertion point of the initialization call.
    auto InitializationInsertPtr = InsertPtr;
    if (LastNonAllocaNonRefReplacement)
      InitializationInsertPtr = LastNonAllocaNonRefReplacement->getNextNode();

    // We now need to insert the sharing calls. We insert after the last value
    // to be replaced or after the alloca.
    if (LastNonAllocaReplacement)
      InsertPtr = LastNonAllocaReplacement->getNextNode();

    if (DSInsertPtr)
      InsertPtr = DSInsertPtr;

    // Do the replacements now.
    for (auto &R : Replacements) {
      auto *From = R.first;
      auto *To = new llvm::LoadInst(R.second, "", /*isVolatile=*/false,
                                    PointerAlign, InsertPtr);
      llvm::Instruction *ToInstr = To;

      // Check if there are uses of From before To and move them after To. These
      // are usually the function epilogue stores.
      for (auto &StartBlock : Fn->getBasicBlockList()) {
        // Only visit the header and omp.init.ds (if it exists) blocks.
        for (auto II = StartBlock.begin(), IE = StartBlock.end(); II != IE;) {
          llvm::Instruction *I = &*II;
          ++II;

          if (I == To)
            break;
          if (I == From)
            continue;

          bool NeedsToMove = false;
          for (auto *U : From->users()) {
            // Is this a user of from? If so we need to move it.
            if (I == U) {
              NeedsToMove = true;
              break;
            }
          }

          if (!NeedsToMove)
            continue;

          I->moveBefore(ToInstr->getNextNode());
          ToInstr = ToInstr->getNextNode();
        }
        // Only visit the header and omp.init.ds (if it exists) blocks.
        if (hasOMPInitDSBlock) {
          if (&StartBlock == DSI.InitDSBlock)
            break;
        } else
            break;
      }

      From->replaceAllUsesWith(To);

      // Make sure the following calls are inserted before these loads.
      InsertPtr = To;
    }

    if (hasOMPInitDSBlock) {
      bool InstructionWasMoved = true;
      while(InstructionWasMoved){
        InstructionWasMoved = false;
        for (auto &StartBlock : Fn->getBasicBlockList()) {
          for (auto II = StartBlock.begin(), IE = StartBlock.end(); II != IE;) {
            llvm::Instruction *OuterInstr = &*II;
            ++II;

            if (isa<llvm::AllocaInst>(OuterInstr))
              continue;

            if (isa<llvm::StoreInst>(OuterInstr))
              continue;

            // Instruction I is the current instruction
            // Scan the two blocks and check if a usage of II is found
            // before it is initialized.
            for (auto *Usage : OuterInstr->users()) {
              // Let's check to see if the usage is BEFORE the init.
              bool InitFound = false;
              for (auto &StartBlock : Fn->getBasicBlockList()) {
                for (auto JJ = StartBlock.begin(), IE = StartBlock.end(); JJ != IE;) {
                  llvm::Instruction *InnerInstr = &*JJ;
                  ++JJ;

                  if (OuterInstr == InnerInstr) {
                    InitFound = true;
                    break;
                  }

                  // PHINodes may reference a value before its initialization but
                  // this might be fine, so don't move it.
                  if (isa<llvm::PHINode>(InnerInstr))
                    continue;

                  if (Usage == InnerInstr)
                    if (!InitFound){
                       InnerInstr->moveBefore(OuterInstr->getNextNode());
                       InstructionWasMoved = true;
                    }

                }
                // Only visit the header and omp.init.ds (if it exists) blocks.
                if (hasOMPInitDSBlock) {
                  if (&StartBlock == DSI.InitDSBlock)
                    break;
                } else
                    break;
              }
            }
          }
          // Only visit the header and omp.init.ds (if it exists) blocks.
          if (hasOMPInitDSBlock) {
            if (&StartBlock == DSI.InitDSBlock)
              break;
          } else
              break;
        }
      }
    }

    if (!hasOMPInitDSBlock) {
      for (auto *I = InsertPtr; I; I = I->getNextNode())
        if (I == InitializationInsertPtr) {
          InitializationInsertPtr = InsertPtr;
          break;
        }
    } else {
      InitializationInsertPtr = &(*DSI.InitDSBlock->begin());
      InsertPtr = InitializationInsertPtr;
    }

    // If this is an entry point, we have to initialize the data sharing first.
    if (DSI.IsEntryPoint)
      InitializeEntryPoint(InitializationInsertPtr);

    // Adjust address spaces in the function arguments.
    auto FArg = DSI.InitializationFunction->arg_begin();
    for (auto &Arg : InitArgs) {
      // If the argument is not in the header of the function (usually because
      // it is after the scheduling of an outermost loop), create a clone
      // in there and use it instead.
      if (auto *I = dyn_cast<llvm::Instruction>(Arg)) {
        if (I->getParent() != &Fn->front()) {
          auto *CI = I->clone();
          Arg = CI;
          CI->insertBefore(InsertPtr);

          if (isa<llvm::LoadInst>(I)) {
            auto LoadedValue = I->getOperand(0);

            for (auto Usage : LoadedValue->users()) {
              if (auto *ST = dyn_cast<llvm::StoreInst>(Usage)) {
                auto *STClone = ST->clone();
                STClone->insertBefore(CI);
                if (auto *LD = dyn_cast<llvm::LoadInst>(STClone->getOperand(0))) {
                  auto *LDClone = LD->clone();
                  STClone->setOperand(0, LDClone);
                  LDClone->insertBefore(STClone);
                }
                break;
              }
            }
          }
        }
      }

      // Types match, nothing to do.
      if (FArg->getType() == Arg->getType()) {
        ++FArg;
        continue;
      }

      // Check if there is some address space mismatch.
      llvm::PointerType *FArgTy = dyn_cast<llvm::PointerType>(FArg->getType());
      llvm::PointerType *ArgTy = dyn_cast<llvm::PointerType>(Arg->getType());
      if (FArgTy && ArgTy &&
          FArgTy->getElementType() == ArgTy->getElementType() &&
          FArgTy->getAddressSpace() != ArgTy->getAddressSpace()) {
        Arg = llvm::CastInst::Create(llvm::CastInst::AddrSpaceCast, Arg, FArgTy,
                                     ".data_share_addrspace_cast", InsertPtr);
        ++FArg;
        continue;
      }

      llvm_unreachable(
          "Unexpected type in data sharing initialization arguments.");
    }

    (void)llvm::CallInst::Create(DSI.InitializationFunction, InitArgs, "",
                                 InsertPtr);

    // Close the environment. The saved stack is in the 4 first entries of the
    // arguments array.
    if (DSI.RequiresOMPRuntime) {
      llvm::Value *ClosingArgs[]{
          InitArgs[0], InitArgs[1], InitArgs[2], InitArgs[3],
          // If an entry point we need to signal the clean up.
          llvm::ConstantInt::get(CGM.Int32Ty, DSI.IsEntryPoint ? 1 : 0)};
      for (llvm::BasicBlock &BB : *Fn)
        if (auto *Ret = dyn_cast<llvm::ReturnInst>(BB.getTerminator()))
          (void)llvm::CallInst::Create(
              createNVPTXRuntimeFunction(
                  OMPRTL_NVPTX__kmpc_data_sharing_environment_end),
              ClosingArgs, "", Ret);
    }
  }

  // Make the default registration procedure.
  return CGOpenMPRuntime::emitRegistrationFunction();
}

StringRef CGOpenMPRuntimeNVPTX::RenameStandardFunction(StringRef name) {
  // Fill up hashmap entries lazily
  if (stdFuncs.empty()) {

    // Trigonometric functions
    stdFuncs.insert(std::make_pair("cos", "__nv_cos"));
    stdFuncs.insert(std::make_pair("sin", "__nv_sin"));
    stdFuncs.insert(std::make_pair("tan", "__nv_tan"));
    stdFuncs.insert(std::make_pair("acos", "__nv_acos"));
    stdFuncs.insert(std::make_pair("asin", "__nv_asin"));
    stdFuncs.insert(std::make_pair("atan", "__nv_atan"));
    stdFuncs.insert(std::make_pair("atan2", "__nv_atan2"));

    stdFuncs.insert(std::make_pair("cosf", "__nv_cosf"));
    stdFuncs.insert(std::make_pair("sinf", "__nv_sinf"));
    stdFuncs.insert(std::make_pair("tanf", "__nv_tanf"));
    stdFuncs.insert(std::make_pair("acosf", "__nv_acosf"));
    stdFuncs.insert(std::make_pair("asinf", "__nv_asinf"));
    stdFuncs.insert(std::make_pair("atanf", "__nv_atanf"));
    stdFuncs.insert(std::make_pair("atan2f", "__nv_atan2f"));

    // Hyperbolic functions
    stdFuncs.insert(std::make_pair("cosh", "__nv_cosh"));
    stdFuncs.insert(std::make_pair("sinh", "__nv_sinh"));
    stdFuncs.insert(std::make_pair("tanh", "__nv_tanh"));
    stdFuncs.insert(std::make_pair("acosh", "__nv_acosh"));
    stdFuncs.insert(std::make_pair("asinh", "__nv_asinh"));
    stdFuncs.insert(std::make_pair("atanh", "__nv_atanh"));

    stdFuncs.insert(std::make_pair("coshf", "__nv_coshf"));
    stdFuncs.insert(std::make_pair("sinhf", "__nv_sinhf"));
    stdFuncs.insert(std::make_pair("tanhf", "__nv_tanhf"));
    stdFuncs.insert(std::make_pair("acoshf", "__nv_acoshf"));
    stdFuncs.insert(std::make_pair("asinhf", "__nv_asinhf"));
    stdFuncs.insert(std::make_pair("atanhf", "__nv_atanhf"));

    // Exponential and logarithm functions
    stdFuncs.insert(std::make_pair("exp", "__nv_exp"));
    stdFuncs.insert(std::make_pair("frexp", "__nv_frexp"));
    stdFuncs.insert(std::make_pair("ldexp", "__nv_ldexp"));
    stdFuncs.insert(std::make_pair("log", "__nv_log"));
    stdFuncs.insert(std::make_pair("log10", "__nv_log10"));
    stdFuncs.insert(std::make_pair("modf", "__nv_modf"));
    stdFuncs.insert(std::make_pair("exp2", "__nv_exp2"));
    stdFuncs.insert(std::make_pair("expm1", "__nv_expm1"));
    stdFuncs.insert(std::make_pair("ilogb", "__nv_ilogb"));
    stdFuncs.insert(std::make_pair("log1p", "__nv_log1p"));
    stdFuncs.insert(std::make_pair("log2", "__nv_log2"));
    stdFuncs.insert(std::make_pair("logb", "__nv_logb"));
    stdFuncs.insert(std::make_pair("scalbn", "__nv_scalbn"));
    //     map.insert(std::make_pair((scalbln", ""));

    stdFuncs.insert(std::make_pair("expf", "__nv_exp"));
    stdFuncs.insert(std::make_pair("frexpf", "__nv_frexpf"));
    stdFuncs.insert(std::make_pair("ldexpf", "__nv_ldexpf"));
    stdFuncs.insert(std::make_pair("logf", "__nv_logf"));
    stdFuncs.insert(std::make_pair("log10f", "__nv_log10f"));
    stdFuncs.insert(std::make_pair("modff", "__nv_modff"));
    stdFuncs.insert(std::make_pair("exp2f", "__nv_exp2f"));
    stdFuncs.insert(std::make_pair("expm1f", "__nv_expm1f"));
    stdFuncs.insert(std::make_pair("ilogbf", "__nv_ilogbf"));
    stdFuncs.insert(std::make_pair("log1pf", "__nv_log1pf"));
    stdFuncs.insert(std::make_pair("log2f", "__nv_log2f"));
    stdFuncs.insert(std::make_pair("logbf", "__nv_logbf"));
    stdFuncs.insert(std::make_pair("scalbnf", "__nv_scalbnf"));
    //     map.insert(std::make_pair("scalblnf", ""));

    // Power functions
    stdFuncs.insert(std::make_pair("pow", "__nv_pow"));
    stdFuncs.insert(std::make_pair("powi", "__nv_powi"));
    stdFuncs.insert(std::make_pair("sqrt", "__nv_sqrt"));
    stdFuncs.insert(std::make_pair("cbrt", "__nv_cbrt"));
    stdFuncs.insert(std::make_pair("hypot", "__nv_hypot"));

    stdFuncs.insert(std::make_pair("powf", "__nv_powf"));
    stdFuncs.insert(std::make_pair("powif", "__nv_powif"));
    stdFuncs.insert(std::make_pair("sqrtf", "__nv_sqrtf"));
    stdFuncs.insert(std::make_pair("cbrtf", "__nv_cbrtf"));
    stdFuncs.insert(std::make_pair("hypotf", "__nv_hypotf"));

    // Error and gamma functions
    stdFuncs.insert(std::make_pair("erf", "__nv_erf"));
    stdFuncs.insert(std::make_pair("erfc", "__nv_erfc"));
    stdFuncs.insert(std::make_pair("tgamma", "__nv_tgamma"));
    stdFuncs.insert(std::make_pair("lgamma", "__nv_lgamma"));

    stdFuncs.insert(std::make_pair("erff", "__nv_erff"));
    stdFuncs.insert(std::make_pair("erfcf", "__nv_erfcf"));
    stdFuncs.insert(std::make_pair("tgammaf", "__nv_tgammaf"));
    stdFuncs.insert(std::make_pair("lgammaf", "__nv_lgammaf"));

    // Rounding and remainder functions
    stdFuncs.insert(std::make_pair("ceil", "__nv_ceil"));
    stdFuncs.insert(std::make_pair("floor", "__nv_floor"));
    stdFuncs.insert(std::make_pair("fmod", "__nv_fmod"));
    stdFuncs.insert(std::make_pair("trunc", "__nv_trunc"));
    stdFuncs.insert(std::make_pair("round", "__nv_round"));
    stdFuncs.insert(std::make_pair("lround", "__nv_lround"));
    stdFuncs.insert(std::make_pair("llround", "__nv_llround"));
    stdFuncs.insert(std::make_pair("rint", "__nv_rint"));
    stdFuncs.insert(std::make_pair("lrint", "__nv_lrint"));
    stdFuncs.insert(std::make_pair("llrint", "__nv_llrint"));
    stdFuncs.insert(std::make_pair("nearbyint", "__nv_nearbyint"));
    stdFuncs.insert(std::make_pair("remainder", "__nv_remainder"));
    stdFuncs.insert(std::make_pair("remquo", "__nv_remquo"));

    stdFuncs.insert(std::make_pair("ceilf", "__nv_ceilf"));
    stdFuncs.insert(std::make_pair("floorf", "__nv_floorf"));
    stdFuncs.insert(std::make_pair("fmodf", "__nv_fmodf"));
    stdFuncs.insert(std::make_pair("truncf", "__nv_truncf"));
    stdFuncs.insert(std::make_pair("roundf", "__nv_roundf"));
    stdFuncs.insert(std::make_pair("lroundf", "__nv_lroundf"));
    stdFuncs.insert(std::make_pair("llroundf", "__nv_llroundf"));
    stdFuncs.insert(std::make_pair("rintf", "__nv_rintf"));
    stdFuncs.insert(std::make_pair("lrintf", "__nv_lrintf"));
    stdFuncs.insert(std::make_pair("llrintf", "__nv_llrintf"));
    stdFuncs.insert(std::make_pair("nearbyintf", "__nv_nearbyintf"));
    stdFuncs.insert(std::make_pair("remainderf", "__nv_remainderf"));
    stdFuncs.insert(std::make_pair("remquof", "__nv_remquof"));

    // Floating-point manipulation functions
    stdFuncs.insert(std::make_pair("copysign", "__nv_copysign"));
    stdFuncs.insert(std::make_pair("nan", "__nv_nan"));
    stdFuncs.insert(std::make_pair("nextafter", "__nv_nextafter"));
    //     map.insert(std::make_pair("nexttoward", ""));

    stdFuncs.insert(std::make_pair("copysignf", "__nv_copysignf"));
    stdFuncs.insert(std::make_pair("nanf", "__nv_nanf"));
    stdFuncs.insert(std::make_pair("nextafterf", "__nv_nextafterf"));
    //     map.insert(std::make_pair("nexttowardf", ""));

    // Minimum, maximu,, difference functions
    stdFuncs.insert(std::make_pair("fdim", "__nv_fdim"));
    stdFuncs.insert(std::make_pair("fmax", "__nv_fmax"));
    stdFuncs.insert(std::make_pair("fmin", "__nv_fmin"));

    stdFuncs.insert(std::make_pair("fdimf", "__nv_fdimf"));
    stdFuncs.insert(std::make_pair("fmaxf", "__nv_fmaxf"));
    stdFuncs.insert(std::make_pair("fminf", "__nv_fminf"));

    // Other functions
    stdFuncs.insert(std::make_pair("fabs", "__nv_fabs"));
    stdFuncs.insert(std::make_pair("abs", "__nv_abs"));
    stdFuncs.insert(std::make_pair("fma", "__nv_fma"));

    stdFuncs.insert(std::make_pair("fabsf", "__nv_fabsf"));
    stdFuncs.insert(std::make_pair("absf", "__nv_absf"));
    stdFuncs.insert(std::make_pair("fmaf", "__nv_fmaf"));

    // temporary solution for Znam: this is how the cuda toolkit defines
    // _Znam but the header file is not properly picked up
    // same for new and delete
    stdFuncs.insert(std::make_pair("_Znam", "malloc"));
    stdFuncs.insert(std::make_pair("_Znwm", "malloc"));
    stdFuncs.insert(std::make_pair("_ZdlPv", "free"));
  }

  // If callee is standard function, change its name
  StringRef match = stdFuncs.lookup(name);
  if (!match.empty()) {
    return match;
  }

  return name;
}

/// This function creates calls to one of two shuffle functions to copy
/// registers between lanes in a warp.
static llvm::Value *CreateRuntimeShuffleFunction(CodeGenFunction &CGF,
                                                 QualType ShuffleTy,
                                                 llvm::Value *Elem,
                                                 llvm::Value *Offset) {
  auto &CGM = CGF.CGM;
  auto &C = CGM.getContext();
  auto &Bld = CGF.Builder;

  unsigned Size = C.getTypeSizeInChars(ShuffleTy).getQuantity();
  assert(Size <= 8 && "Unsupported bitwidth in shuffle instruction.");

  llvm::Constant *RTLFn = nullptr;
  if (Size <= 4) {
    // Build int32 __kmpc_shuffle_long(int32 elem, int16 offset, int16
    // num_participants);
    llvm::Type *TypeParams[] = {CGM.Int32Ty, CGM.Int16Ty, CGM.Int16Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_shuffle_int32");
  } else {
    // Build double __kmpc_shuffle_long_long(int64 elem, int16 offset, int16
    // num_participants);
    llvm::Type *TypeParams[] = {CGM.Int64Ty, CGM.Int16Ty, CGM.Int16Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int64Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_shuffle_int64");
  }

  // Cast all types to 32 or 64-bit values before calling shuffle routines.
  auto CastTy = Size <= 4 ? llvm::Type::getInt32Ty(CGM.getLLVMContext())
                          : llvm::Type::getInt64Ty(CGM.getLLVMContext());
  Elem = Bld.CreateSExtOrBitCast(Elem, CastTy);

  llvm::SmallVector<llvm::Value *, 3> FnArgs;
  FnArgs.push_back(Elem);
  FnArgs.push_back(Offset);
  FnArgs.push_back(Bld.getInt16(DS_Max_Worker_Warp_Size));

  llvm::Value *ShuffledVal = CGF.EmitNounwindRuntimeCall(RTLFn, FnArgs);
  return Bld.CreateTruncOrBitCast(ShuffledVal,
                                  CGF.ConvertTypeForMem(ShuffleTy));
}

static void CopyReductionVariable(
    CodeGenFunction &CGF, QualType Ty, Address Src, Address Dest,
    const llvm::function_ref<void(QualType Ty, Address, Address)>
        &ScalarCopyMaker,
    const llvm::function_ref<void(unsigned, llvm::Value *, llvm::Value *,
                                  llvm::Value *)> &ArrayElementCopyMaker,
    SourceLocation Loc) {
  if (Ty->isScalarType()) {
    ScalarCopyMaker(Ty, Src, Dest);
  } else {
    auto &&LoopGen = [&CGF, Loc](
        unsigned LoopLB, unsigned LoopUB,
        const llvm::function_ref<void(unsigned, llvm::Value *, llvm::Value *,
                                      llvm::Value *)> &CopyMaker,
        unsigned ElementSize, llvm::Value *Src, llvm::Value *Dest) {
      auto &Bld = CGF.Builder;
      llvm::BasicBlock *LoopBB = CGF.createBasicBlock("copy.loop");
      llvm::BasicBlock *ContBB = CGF.createBasicBlock("copy.loop.end");

      auto Int64Ty = CGF.CGM.getContext().getIntTypeForBitwidth(
          /*DestWidth=*/64, /*Signed=*/false);
      Address Idx = CGF.CreateMemTemp(Int64Ty, "loop.idx");
      Address UB = CGF.CreateMemTemp(Int64Ty, "loop.ub");
      CGF.EmitStoreOfScalar(Bld.getInt64(LoopLB), Idx, /*Volatile=*/false,
                            Int64Ty);
      CGF.EmitStoreOfScalar(Bld.getInt64(LoopUB), UB, /*Volatile=*/false,
                            Int64Ty);

      CGF.EmitBlock(LoopBB);
      llvm::Value *IdxVal =
          CGF.EmitLoadOfScalar(Idx, /*Volatile=*/false, Int64Ty, Loc);
      CopyMaker(ElementSize, Src, Dest, IdxVal);

      llvm::Value *IdxIncr = Bld.CreateNUWAdd(IdxVal, Bld.getInt64(1));
      CGF.EmitStoreOfScalar(IdxIncr, Idx, /*Volatile=*/false, Int64Ty);
      llvm::Value *UBVal =
          CGF.EmitLoadOfScalar(UB, /*Volatile=*/false, Int64Ty, Loc);
      llvm::Value *AtEnd = Bld.CreateICmpEQ(IdxIncr, UBVal, "copy.loop.atend");
      Bld.CreateCondBr(AtEnd, ContBB, LoopBB);

      CGF.EmitBlock(ContBB);
    };
    auto &Bld = CGF.Builder;
    unsigned Bytes = CGF.getContext().getTypeSizeInChars(Ty).getQuantity();
    unsigned Words = Bytes / 8;
    // Shuffle chunks of 64-bit words.
    if (Words > 0) {
      auto *ArrayTy = CGF.CGM.Int64Ty->getPointerTo();
      auto *SrcCast = Bld.CreatePointerCast(Src.getPointer(), ArrayTy);
      auto *DestCast = Bld.CreatePointerCast(Dest.getPointer(), ArrayTy);
      LoopGen(0, Words, ArrayElementCopyMaker, /*ElementSize=*/64, SrcCast,
              DestCast);
    }

    // Shuffle one remainder of 32 bits.
    unsigned Remainder = Bytes % 8;
    if (Remainder >= 4) {
      unsigned HalfWords = Bytes / 4;
      auto *ArrayTy = CGF.CGM.Int32Ty->getPointerTo();
      auto *SrcCast = Bld.CreatePointerCast(Src.getPointer(), ArrayTy);
      auto *DestCast = Bld.CreatePointerCast(Dest.getPointer(), ArrayTy);
      ArrayElementCopyMaker(/*ElementSize=*/32, SrcCast, DestCast,
                            Bld.getInt64(HalfWords - 1));
      Remainder -= 4;
    }

    // Shuffle one remainder of 16 bits.
    if (Remainder >= 2) {
      unsigned ShortWords = Bytes / 2;
      auto *ArrayTy = CGF.CGM.Int16Ty->getPointerTo();
      auto *SrcCast = Bld.CreatePointerCast(Src.getPointer(), ArrayTy);
      auto *DestCast = Bld.CreatePointerCast(Dest.getPointer(), ArrayTy);
      ArrayElementCopyMaker(/*ElementSize=*/16, SrcCast, DestCast,
                            Bld.getInt64(ShortWords - 1));
      Remainder -= 2;
    }

    // Shuffle one remainder of 8 bits.
    if (Remainder) {
      auto *ArrayTy = CGF.CGM.Int8Ty->getPointerTo();
      auto *SrcCast = Bld.CreatePointerCast(Src.getPointer(), ArrayTy);
      auto *DestCast = Bld.CreatePointerCast(Dest.getPointer(), ArrayTy);
      ArrayElementCopyMaker(/*ElementSize=*/8, SrcCast, DestCast,
                            Bld.getInt64(Bytes - 1));
    }
  }
}

static void EmitDirectionSpecializedReduceDataCopy(
    COPY_DIRECTION Direction, CodeGenFunction &CGF, QualType ReductionArrayTy,
    ArrayRef<const Expr *> Privates, Address SrcBase, Address DestBase,
    llvm::Value *OffsetVal = nullptr, llvm::Value *IndexVal = nullptr,
    llvm::Value *WidthVal = nullptr) {

  auto &CGM = CGF.CGM;
  auto &C = CGM.getContext();
  auto &Bld = CGF.Builder;

  // In this section, we distinguish between two concepts, ReduceData Element
  // and ReduceData Subelement:
  // 1. ReduceData Element refers to an object directly referenced by a field
  // in ReduceData, this could be a built-in type such as int32 or a constant
  // length array type [4 x int32]
  // 2. ReduceData Subelement refers to an object of built-in type with
  // maximum bit size of 64. It could be an array element of a
  // constant-length-array-typed ReduceData Element.

  // This loop iterates through the list of reduce elements and copies element
  // by element from a remote lane to the local variable remote_red_list.
  unsigned Idx = 0;
  for (auto &Private : Privates) {
    Address SrcElementAddr = Address::invalid();
    Address DestElementAddr = Address::invalid();
    Address DestElementPtrAddr = Address::invalid();

    unsigned ElementSizeInChars = 0;
    switch (Direction) {
    case Shuffle_To_ReduceData: {
      // Step 1.1 get the address for src element.
      Address SrcElementPtrAddr =
          Bld.CreateConstArrayGEP(SrcBase, Idx, CGF.getPointerSize());
      llvm::Value *SrcElementPtrPtr =
          CGF.EmitLoadOfScalar(SrcElementPtrAddr, /*Volatile=*/false,
                               C.VoidPtrTy, Private->getExprLoc());
      SrcElementAddr =
          Address(SrcElementPtrPtr, C.getTypeAlignInChars(Private->getType()));

      // Step 1.2 get the address for dest element this is a temporary memory
      // element.
      DestElementPtrAddr =
          Bld.CreateConstArrayGEP(DestBase, Idx, CGF.getPointerSize());
      DestElementAddr =
          CGF.CreateMemTemp(Private->getType(), ".omp.reduction.element");
      break;
    }
    case ReduceData_To_ReduceData: {
      // Step 1.1 get the address for src element.
      Address SrcElementPtrAddr =
          Bld.CreateConstArrayGEP(SrcBase, Idx, CGF.getPointerSize());
      llvm::Value *SrcElementPtrPtr =
          CGF.EmitLoadOfScalar(SrcElementPtrAddr, /*Volatile=*/false,
                               C.VoidPtrTy, Private->getExprLoc());
      SrcElementAddr =
          Address(SrcElementPtrPtr, C.getTypeAlignInChars(Private->getType()));

      // Step 1.2 get the address for dest element.
      DestElementPtrAddr =
          Bld.CreateConstArrayGEP(DestBase, Idx, CGF.getPointerSize());
      llvm::Value *DestElementPtr =
          CGF.EmitLoadOfScalar(DestElementPtrAddr, /*Volatile=*/false,
                               C.VoidPtrTy, Private->getExprLoc());
      Address DestElemAddr =
          Address(DestElementPtr, C.getTypeAlignInChars(Private->getType()));
      DestElementAddr = Bld.CreateElementBitCast(
          DestElemAddr, CGF.ConvertTypeForMem(Private->getType()));
      break;
    }
    case ReduceData_To_Global: {
      // Step 1.1 get the address for src element.
      Address SrcElementPtrAddr =
          Bld.CreateConstArrayGEP(SrcBase, Idx, CGF.getPointerSize());
      llvm::Value *SrcElementPtrPtr =
          CGF.EmitLoadOfScalar(SrcElementPtrAddr, /*Volatile=*/false,
                               C.VoidPtrTy, Private->getExprLoc());
      SrcElementAddr =
          Address(SrcElementPtrPtr, C.getTypeAlignInChars(Private->getType()));

// Step 1.2 get the address for dest element by
// address = base + index * ElementSizeInChars.
#if 0
      unsigned BitSize =
          CGF.ConvertTypeForMem(Private->getType())->getPrimitiveSizeInBits();
      if (!BitSize)
        BitSize = getArrayTypeSize(CGF.ConvertTypeForMem(Private->getType()));
      ElementSizeInChars = BitSize / 8;
#endif
      ElementSizeInChars =
          C.getTypeSizeInChars(Private->getType()).getQuantity();
      auto *CurrentOffset =
          Bld.CreateMul(Bld.getInt32(ElementSizeInChars), IndexVal);
      CurrentOffset = Bld.CreateSExt(CurrentOffset, CGF.Int64Ty);
      auto *ScratchPadElemAbsolutePtrVal =
          Bld.CreateAdd(DestBase.getPointer(), CurrentOffset);
      ScratchPadElemAbsolutePtrVal =
          Bld.CreateIntToPtr(ScratchPadElemAbsolutePtrVal, CGF.VoidPtrTy);
      Address ScratchpadPtr =
          Address(ScratchPadElemAbsolutePtrVal,
                  C.getTypeAlignInChars(Private->getType()));
      DestElementAddr = Bld.CreateElementBitCast(
          ScratchpadPtr, CGF.ConvertTypeForMem(Private->getType()));
      break;
    }
    case Global_To_ReduceData: {
// Step 1.1 get the address for src element address by
// address = base + index * ElementSizeInChars.
#if 0
      unsigned BitSize =
          CGF.ConvertTypeForMem(Private->getType())->getPrimitiveSizeInBits();
      if (!BitSize)
        BitSize = getArrayTypeSize(CGF.ConvertTypeForMem(Private->getType()));
      ElementSizeInChars = BitSize / 8;
#endif
      ElementSizeInChars =
          C.getTypeSizeInChars(Private->getType()).getQuantity();
      auto *CurrentOffset =
          Bld.CreateMul(Bld.getInt32(ElementSizeInChars), IndexVal);
      CurrentOffset = Bld.CreateSExt(CurrentOffset, CGF.Int64Ty);
      auto *ScratchPadElemAbsolutePtrVal =
          Bld.CreateAdd(SrcBase.getPointer(), CurrentOffset);
      ScratchPadElemAbsolutePtrVal =
          Bld.CreateIntToPtr(ScratchPadElemAbsolutePtrVal, CGF.VoidPtrTy);
      SrcElementAddr = Address(ScratchPadElemAbsolutePtrVal,
                               C.getTypeAlignInChars(Private->getType()));

      // Step 1.2 get the address for dest element this is a temporary memory
      // element.
      DestElementPtrAddr =
          Bld.CreateConstArrayGEP(DestBase, Idx, CGF.getPointerSize());
      DestElementAddr =
          CGF.CreateMemTemp(Private->getType(), ".omp.reduction.element");
      break;
    }
    }

    // Regardless of src and dest of memory copy, we emit the load of src
    // element as this is required in all directions
    SrcElementAddr = Bld.CreateElementBitCast(
        SrcElementAddr, CGF.ConvertTypeForMem(Private->getType()));

    if (Direction == Shuffle_To_ReduceData) {
      // Shuffle a variable from another lane.
      auto &&ScalarCopyMaker = [&CGF, OffsetVal, Private](
                                   QualType Ty, Address Src, Address Dest) {
        llvm::Value *Elem = CGF.EmitLoadOfScalar(Src, /*Volatile=*/false, Ty,
                                                 Private->getExprLoc());
        llvm::Value *ShuffledVal =
            CreateRuntimeShuffleFunction(CGF, Ty, Elem, OffsetVal);
        CGF.EmitStoreOfScalar(ShuffledVal, Dest, /*Volatile=*/false, Ty);
      };
      auto &&ArrayElementCopyMaker = [&CGF, &C, &Bld, OffsetVal, Private](
          unsigned ElementSize, llvm::Value *Src, llvm::Value *Dest,
          llvm::Value *Idx) {
        auto ShuffleTy = C.getIntTypeForBitwidth(/*DestWidth*/ ElementSize,
                                                 /*Signed*/ false);
        Address SrcElementPtr(Bld.CreateGEP(Src, Idx),
                              C.getTypeAlignInChars(ShuffleTy));
        llvm::Value *SrcElement =
            CGF.EmitLoadOfScalar(SrcElementPtr, /*Volatile=*/false, ShuffleTy,
                                 Private->getExprLoc());
        llvm::Value *ShuffledVal =
            CreateRuntimeShuffleFunction(CGF, ShuffleTy, SrcElement, OffsetVal);
        Address DestElementPtr(Bld.CreateGEP(Dest, Idx),
                               C.getTypeAlignInChars(ShuffleTy));
        CGF.EmitStoreOfScalar(ShuffledVal, DestElementPtr, /*Volatile=*/false,
                              ShuffleTy);
      };
      CopyReductionVariable(CGF, Private->getType(), SrcElementAddr,
                            DestElementAddr, ScalarCopyMaker,
                            ArrayElementCopyMaker, Private->getExprLoc());
    } else {
      // Copy a variable from memory.  Use 'memcpy' for structures.
      if (Private->getType()->isScalarType()) {
        llvm::Value *Elem =
            CGF.EmitLoadOfScalar(SrcElementAddr, /*Volatile=*/false,
                                 Private->getType(), Private->getExprLoc());
        // Just store the element value we have already obtained to dest element
        // address.
        CGF.EmitStoreOfScalar(Elem, DestElementAddr, /*Volatile=*/false,
                              Private->getType());
      } else {
        unsigned Bytes = CGF.getContext()
                             .getTypeSizeInChars(Private->getType())
                             .getQuantity();
        Bld.CreateMemCpy(DestElementAddr, SrcElementAddr, Bytes);
      }
    }

#if 0
// Step 2.1 create individual load (potentially with shuffle) and stores
// for each ReduceData Element and ReduceData subelement.

#if 0
    // If we have an array, it must be a constant length array.
    if (llvm::ArrayType *ElemArrayType =
            dyn_cast<llvm::ArrayType>(Elem->getType())) {
      // FIXME: Emit loop
      for (uint64_t ArrayIndex = 0;
           ArrayIndex < ElemArrayType->getNumElements(); ArrayIndex++) {
        // Get the pointer to src and dest subelement.
        auto RemoteArrayCopyBaseAddr = DestElementAddr;
        auto LocalArrayBaseAddr = SrcElementAddr;
        auto RemoteArrayCopyAddrVal = Bld.CreateInBoundsGEP(
            RemoteArrayCopyBaseAddr.getPointer(),
            {Bld.getInt64(0), Bld.getInt64(ArrayIndex)});
        auto LocalArrayAddrVal = Bld.CreateInBoundsGEP(
            LocalArrayBaseAddr.getPointer(),
            {Bld.getInt64(0), Bld.getInt64(ArrayIndex)});
        Address RemoteArrayCopyAddr = Address(
            RemoteArrayCopyAddrVal, C.getTypeAlignInChars(Private->getType()));
        Address LocalArrayAddr = Address(
            LocalArrayAddrVal, C.getTypeAlignInChars(Private->getType()));

        // load src sub element
        llvm::Value *SubElem = Bld.CreateLoad(LocalArrayAddr);

        // We have to shuffle SubElem to get the actual data we want for
        // Shuffle_To_ReduceData Direction
        if (Direction == Shuffle_To_ReduceData) {
          llvm::Value *ShuffledVal = CreateRuntimeShuffleFunction(
              CGF, ElemArrayType->getElementType(), SubElem, OffsetVal);
          SubElem = Bld.CreateTruncOrBitCast(
              ShuffledVal, ElemArrayType->getElementType());
        }

        // Store to dest subelement address.
        Bld.CreateStore(SubElem, RemoteArrayCopyAddr);
      }
      // Else we don't have an array-typed element.
    } else {
#endif
    if (Direction == Shuffle_To_ReduceData) {
      llvm::Value *ShuffledVal = CreateRuntimeShuffleFunction(
          CGF, Private->getType(), Elem, OffsetVal);
      Elem = Bld.CreateTruncOrBitCast(
          ShuffledVal, CGF.ConvertTypeForMem(Private->getType()));
    }

    // Just store the element value we have already obtained to dest element
    // address.
    CGF.EmitStoreOfScalar(Elem, DestElementAddr, /*Volatile=*/false,
                          Private->getType());
#if 0
    }
#endif
#endif

    // Step 3.1 modify reference in Dest ReduceData as needed.
    if (Direction == Shuffle_To_ReduceData ||
        Direction == Global_To_ReduceData) {
      // Here we are modifying the reference directly in ReduceData because
      // we are creating a local temporary copy of remote ReduceData.
      // The variable is only alive in the current function scope and the
      // scope of functions it invokes (i.e., reduce_function)
      // RemoteReduceData[i] = (void*)&RemoteElem
      CGF.EmitStoreOfScalar(Bld.CreatePointerBitCastOrAddrSpaceCast(
                                DestElementAddr.getPointer(), CGF.VoidPtrTy),
                            DestElementPtrAddr, /*Volatile=*/false,
                            C.VoidPtrTy);
    }

    if (Direction == ReduceData_To_Global ||
        Direction == Global_To_ReduceData) {
      // Step 4.1 increment SrcBase/DestBase so that it points to the starting
      // address of the next element in scratchpad memory. Memory alignment is
      // also taken care of in this step.
      llvm::Value *DestOrSrcBasePtrVal = Direction == ReduceData_To_Global
                                             ? DestBase.getPointer()
                                             : SrcBase.getPointer();
      DestOrSrcBasePtrVal = Bld.CreateAdd(
          DestOrSrcBasePtrVal,
          Bld.CreateMul(WidthVal, Bld.getInt64(ElementSizeInChars)));

      // Take care of 256 byte alignment
      DestOrSrcBasePtrVal = Bld.CreateSub(DestOrSrcBasePtrVal, Bld.getInt64(1));
      DestOrSrcBasePtrVal =
          Bld.CreateSDiv(DestOrSrcBasePtrVal, Bld.getInt64(256));
      DestOrSrcBasePtrVal = Bld.CreateAdd(DestOrSrcBasePtrVal, Bld.getInt64(1));
      DestOrSrcBasePtrVal =
          Bld.CreateMul(DestOrSrcBasePtrVal, Bld.getInt64(256));

      if (Direction == ReduceData_To_Global)
        DestBase = Address(DestOrSrcBasePtrVal, CGF.getPointerAlign());
      else /* Direction == Global_To_ReduceData */
        SrcBase = Address(DestOrSrcBasePtrVal, CGF.getPointerAlign());
    }

    Idx++;
  }
}

llvm::Value *EmitCopyToScratchpad(CodeGenModule &CGM,
                                  ArrayRef<const Expr *> Privates,
                                  QualType ReductionArrayTy,
                                  SourceLocation Loc) {

  //
  //  for elem in ReduceData:
  //    scratchpad[elem_id][index] = ReduceData.elem
  //
  auto &C = CGM.getContext();

  FunctionArgList Args;

  // ReduceData- this is the source of the copy.
  ImplicitParamDecl ReduceDataArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                  C.VoidPtrTy, ImplicitParamDecl::Other);
  // This is the pointer to the scratchpad array, with each element
  // storing ReduceData.
  ImplicitParamDecl ScratchPadArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                  C.VoidPtrTy, ImplicitParamDecl::Other);
  // This argument specifies the index into the scratchpad array,
  // typically the TeamId.
  ImplicitParamDecl IndexArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.IntTy,
                             ImplicitParamDecl::Other);
  // This argument specifies the row width of an element, typically
  // the number of teams.
  ImplicitParamDecl WidthArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.IntTy,
                             ImplicitParamDecl::Other);
  Args.push_back(&ReduceDataArg);
  Args.push_back(&ScratchPadArg);
  Args.push_back(&IndexArg);
  Args.push_back(&WidthArg);

  auto &CGFI = CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      ".omp.reduction.copy_to_scratchpad", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*DC=*/nullptr, Fn, CGFI);
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, CGFI, Args, Loc, Loc);

  auto &Bld = CGF.Builder;

  // Get ReduceData as a function parameter.
  Address AddrReduceDataArg = CGF.GetAddrOfLocalVar(&ReduceDataArg);
  Address SrcDataAddr(
      Bld.CreatePointerBitCastOrAddrSpaceCast(
          CGF.EmitLoadOfScalar(AddrReduceDataArg, /*Volatile=*/false,
                               C.VoidPtrTy, Loc),
          CGF.ConvertTypeForMem(ReductionArrayTy)->getPointerTo()),
      CGF.getPointerAlign());

  // Get ScratchPad pointer.
  Address AddrScratchPadArg = CGF.GetAddrOfLocalVar(&ScratchPadArg);
  llvm::Value *ScratchPadVal = CGF.EmitLoadOfScalar(
      AddrScratchPadArg, /*Volatile=*/false, C.VoidPtrTy, Loc);

  // Get Index value
  Address AddrIndexArg = CGF.GetAddrOfLocalVar(&IndexArg);
  llvm::Value *IndexVal =
      CGF.EmitLoadOfScalar(AddrIndexArg, /*Volatile=*/false, C.IntTy, Loc);

  // Get Width of the scratchpad array (number of teams)
  Address AddrWidthArg = CGF.GetAddrOfLocalVar(&WidthArg);
  llvm::Value *WidthVal = Bld.CreateSExt(
      CGF.EmitLoadOfScalar(AddrWidthArg, /*Volatile=*/false, C.IntTy, Loc),
      CGF.Int64Ty);

  // The absolute ptr address to the base addr of the next element to copy
  // CumulativeElemBasePtr = &Scratchpad[some element][0]
  // convert to 64 bit int for pointer calculation
  llvm::Value *CumulativeElemBasePtr =
      Bld.CreatePtrToInt(ScratchPadVal, CGM.Int64Ty);
  Address DestDataAddr(CumulativeElemBasePtr, CGF.getPointerAlign());

  EmitDirectionSpecializedReduceDataCopy(
      ReduceData_To_Global, CGF, ReductionArrayTy, Privates, SrcDataAddr,
      DestDataAddr, nullptr, IndexVal, WidthVal);

  CGF.FinishFunction();
  return Fn;
}

static llvm::Value *EmitReduceScratchpadFunction(
    CodeGenModule &CGM, ArrayRef<const Expr *> Privates,
    QualType ReductionArrayTy, llvm::Value *ReduceFn, SourceLocation Loc) {
  auto &C = CGM.getContext();
  //
  //  load_and_reduce(local, scratchpad, index, width, reduce)
  //  ReduceData remote;
  //  for elem in ReduceData:
  //    remote.elem = Scratchpad[elem_id][index]
  //  if (reduce)
  //    local = local @ remote
  //  else
  //    local = remote
  //
  FunctionArgList Args;

  // This is the pointer that points to ReduceData.
  ImplicitParamDecl ReduceDataArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                  C.VoidPtrTy, ImplicitParamDecl::Other);
  // Pointer to the scratchpad.
  ImplicitParamDecl ScratchPadArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                  C.VoidPtrTy, ImplicitParamDecl::Other);
  // This argument specifies the index of the ReduceData in the
  // scratchpad.
  ImplicitParamDecl IndexArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.IntTy,
                             ImplicitParamDecl::Other);
  // This argument specifies the row width of an element.
  ImplicitParamDecl WidthArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.IntTy,
                             ImplicitParamDecl::Other);
  // If should_reduce == 1, then it's load AND reduce,
  // If should_reduce == 0 (or otherwise), then it only loads (+ copy).
  ImplicitParamDecl ShouldReduceArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                    C.IntTy, ImplicitParamDecl::Other);

  Args.push_back(&ReduceDataArg);
  Args.push_back(&ScratchPadArg);
  Args.push_back(&IndexArg);
  Args.push_back(&WidthArg);
  Args.push_back(&ShouldReduceArg);

  auto &CGFI = CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      ".omp.reduction.load_and_reduce", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*DC=*/nullptr, Fn, CGFI);
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, CGFI, Args, Loc, Loc);

  auto &Bld = CGF.Builder;

  // Get local ReduceData pointer.
  Address AddrReduceDataArg = CGF.GetAddrOfLocalVar(&ReduceDataArg);
  Address ReduceDataAddr(
      Bld.CreatePointerBitCastOrAddrSpaceCast(
          CGF.EmitLoadOfScalar(AddrReduceDataArg, /*Volatile=*/false,
                               C.VoidPtrTy, Loc),
          CGF.ConvertTypeForMem(ReductionArrayTy)->getPointerTo()),
      CGF.getPointerAlign());
  // Get ScratchPad pointer.
  Address AddrScratchPadArg = CGF.GetAddrOfLocalVar(&ScratchPadArg);
  llvm::Value *ScratchPadVal = CGF.EmitLoadOfScalar(
      AddrScratchPadArg, /*Volatile=*/false, C.VoidPtrTy, Loc);
  // Get Index value
  Address AddrIndexArg = CGF.GetAddrOfLocalVar(&IndexArg);
  llvm::Value *IndexVal =
      CGF.EmitLoadOfScalar(AddrIndexArg, /*Volatile=*/false, C.IntTy, Loc);
  // Get Width of the scratchpad array (number of teams)
  Address AddrWidthArg = CGF.GetAddrOfLocalVar(&WidthArg);
  llvm::Value *WidthVal = Bld.CreateSExt(
      CGF.EmitLoadOfScalar(AddrWidthArg, /*Volatile=*/false, C.IntTy, Loc),
      CGF.Int64Ty);
  // Get whether-to-reduce flag
  Address AddrShouldReduceArg = CGF.GetAddrOfLocalVar(&ShouldReduceArg);
  llvm::Value *ShouldReduceVal = CGF.EmitLoadOfScalar(
      AddrShouldReduceArg, /*Volatile=*/false, C.IntTy, Loc);

  // The absolute ptr address to the base addr of the next element to copy
  // CumulativeElemBasePtr = &Scratchpad[some element][0]
  // convert to 64 bit int for pointer calculation
  llvm::Value *CumulativeElemBasePtr =
      Bld.CreatePtrToInt(ScratchPadVal, CGM.Int64Ty);
  Address SrcDataAddr(CumulativeElemBasePtr, CGF.getPointerAlign());

  // create remote ReduceData pointer
  Address RemoteReduceData =
      CGF.CreateMemTemp(ReductionArrayTy, ".omp.reduction.remote_red_list");

  // Assemble RemoteReduceData.
  EmitDirectionSpecializedReduceDataCopy(
      Global_To_ReduceData, CGF, ReductionArrayTy, Privates, SrcDataAddr,
      RemoteReduceData, nullptr, IndexVal, WidthVal);

  llvm::BasicBlock *ThenBB = CGF.createBasicBlock("then");
  llvm::BasicBlock *ElseBB = CGF.createBasicBlock("else");
  llvm::BasicBlock *MergeBB = CGF.createBasicBlock("ifcont");

  // Do we want to reduce?
  auto CondReduce = Bld.CreateICmpEQ(ShouldReduceVal, Bld.getInt32(1));
  Bld.CreateCondBr(CondReduce, ThenBB, ElseBB);

  CGF.EmitBlock(ThenBB);
  // If yes, we want to reduce, do the reduce
  // by calling ReduceFn
  llvm::SmallVector<llvm::Value *, 2> FnArgs;
  llvm::Value *LocalDataPtr = Bld.CreatePointerBitCastOrAddrSpaceCast(
      ReduceDataAddr.getPointer(), CGF.VoidPtrTy);
  FnArgs.push_back(LocalDataPtr);
  llvm::Value *RemoteDataPtr = Bld.CreatePointerBitCastOrAddrSpaceCast(
      RemoteReduceData.getPointer(), CGF.VoidPtrTy);
  FnArgs.push_back(RemoteDataPtr);
  CGM.getOpenMPRuntime().emitOutlinedFunctionCall(CGF, Loc, ReduceFn, FnArgs);
  Bld.CreateBr(MergeBB);

  // Else no, just copy
  // localReduceData = RemoteReduceData.
  CGF.EmitBlock(ElseBB);
  EmitDirectionSpecializedReduceDataCopy(ReduceData_To_ReduceData, CGF,
                                         ReductionArrayTy, Privates,
                                         RemoteReduceData, ReduceDataAddr);
  Bld.CreateBr(MergeBB);

  // endif
  CGF.EmitBlock(MergeBB);
  CGF.FinishFunction();
  return Fn;
}

static llvm::Value *EmitInterWarpCopyFunction(CodeGenModule &CGM,
                                              ArrayRef<const Expr *> Privates,
                                              QualType ReductionArrayTy,
                                              SourceLocation Loc) {
  //
  // This function emits code that gathers reduce_data from the first lane of
  // every active warp to lanes in the first warp.
  // void inter_warp_copy_func(void* reduce_data, warp_num)
  // shared smem[32];
  // For all data entries D in reduce_data:
  //    If (I am the first lane in each warp)
  //      Copy my local D to smem[warp_id]
  //    sync
  //    if (I am the first warp)
  //      Copy smem[thread_id] to my local D
  //    sync
  //

  auto &C = CGM.getContext();
  auto &M = CGM.getModule();

  // ReduceData: thread local reduce_data.
  // At the stage of the computation when this function is called,
  // useful values reside in the first lane of every warp.
  ImplicitParamDecl ReduceDataArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                  C.VoidPtrTy, ImplicitParamDecl::Other);
  // WarpNum: number of warps needed to compute the entire block
  // of elements. This could be smaller than 32 for partial
  // block reduction.
  ImplicitParamDecl WarpNumArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.IntTy,
                               ImplicitParamDecl::Other);
  FunctionArgList Args;
  Args.push_back(&ReduceDataArg);
  Args.push_back(&WarpNumArg);
  auto &CGFI = CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      ".omp.reduction.inter_warp_copy_func", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*DC=*/nullptr, Fn, CGFI);
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, CGFI, Args, Loc, Loc);

  auto &Bld = CGF.Builder;

  // This array is used as a medium to transfer, one reduce element at a time,
  // the data from the first lane of every warp to lanes in the first warp
  // in order to perform the final step of block reduction (reduction across
  // warps)
  // It is declared with common linkage so as to be shared across
  // compilation units.
  const char *Name = "__openmp_nvptx_data_transfer_temporary_storage";
  llvm::GlobalVariable *Gbl = M.getGlobalVariable(Name);
  if (!Gbl) {
    auto *Ty = llvm::ArrayType::get(CGM.Int64Ty, /*warpSize=*/32);
    Gbl = new llvm::GlobalVariable(
        M, Ty,
        /*isConstant=*/false, llvm::GlobalVariable::CommonLinkage,
        llvm::Constant::getNullValue(Ty), Name,
        /*InsertBefore=*/nullptr, llvm::GlobalVariable::NotThreadLocal,
        /*AddressSpace=Shared*/ ADDRESS_SPACE_SHARED);
  }

  // Get the id of the current thread on the GPU.
  auto *ThreadId = GetNVPTXThreadID(CGF);
  // nvptx_lane_id = nvptx_id % 32
  auto *LaneId = GetNVPTXThreadWarpID(CGF);
  // nvptx_warp_id = nvptx_id / 32
  auto *WarpId = GetNVPTXWarpID(CGF);
  // get WarpNumNeeded as a function parameter
  Address AddrWarpNumArg = CGF.GetAddrOfLocalVar(&WarpNumArg);
  llvm::Value *WarpNumVal =
      CGF.EmitLoadOfScalar(AddrWarpNumArg, /*Volatile=*/false, C.IntTy, Loc);
  // SrcDataAddr = <reduce_data_type> (*reduce_data)
  Address AddrReduceDataArg = CGF.GetAddrOfLocalVar(&ReduceDataArg);
  Address SrcDataAddr(
      Bld.CreatePointerBitCastOrAddrSpaceCast(
          CGF.EmitLoadOfScalar(AddrReduceDataArg, /*Volatile=*/false,
                               C.VoidPtrTy, Loc),
          CGF.ConvertTypeForMem(ReductionArrayTy)->getPointerTo()),
      CGF.getPointerAlign());

  unsigned Idx = 0;
  for (auto &Private : Privates) {
    // elemptrptr =SrcDataAddr[i]
    Address ElemPtrPtrAddr =
        Bld.CreateConstArrayGEP(SrcDataAddr, Idx, CGF.getPointerSize());
    llvm::Value *ElemPtrPtr = CGF.EmitLoadOfScalar(
        ElemPtrPtrAddr, /*Volatile=*/false, C.VoidPtrTy, Loc);
    // elemptr = (type[i]*)(elemptrptr)
    Address ElemPtr =
        Address(ElemPtrPtr, C.getTypeAlignInChars(Private->getType()));
    ElemPtr = Bld.CreateElementBitCast(
        ElemPtr, CGF.ConvertTypeForMem(Private->getType()));

    // TargetElemPtr = (type[i]*)(SrcDataAddr[i])
    Address TargetElemPtrPtr =
        Bld.CreateConstArrayGEP(SrcDataAddr, Idx, CGF.getPointerSize());
    llvm::Value *TargetElemPtrVal = CGF.EmitLoadOfScalar(
        TargetElemPtrPtr, /*Volatile=*/false, C.VoidPtrTy, Loc);
    Address TargetElemPtr =
        Address(TargetElemPtrVal, C.getTypeAlignInChars(Private->getType()));
    TargetElemPtr = Bld.CreateElementBitCast(
        TargetElemPtr, CGF.ConvertTypeForMem(Private->getType()));

    auto &&CopyThruSharedMemory = [&CGF, &CGM, &C, &Bld, Gbl, ThreadId, LaneId,
                                   WarpId, WarpNumVal, Loc](
                                      QualType Ty,
                                      const llvm::function_ref<llvm::Value *()>
                                          &LoadScalarSrc,
                                      const llvm::function_ref<void(
                                          llvm::Value *)> &StoreScalarTarget) {
      llvm::BasicBlock *ThenBB = CGF.createBasicBlock("then");
      llvm::BasicBlock *ElseBB = CGF.createBasicBlock("else");
      llvm::BasicBlock *MergeBB = CGF.createBasicBlock("ifcont");

      // if (lane_id == 0)
      auto IsWarpMaster = Bld.CreateICmpEQ(LaneId, Bld.getInt32(0), "ifcond");
      Bld.CreateCondBr(IsWarpMaster, ThenBB, ElseBB);
      CGF.EmitBlock(ThenBB);

      // elem = *elemptr
      llvm::Value *Elem = LoadScalarSrc();

      // MediumPtr = &shared[warp_id]
      llvm::Value *MediumPtrVal = Bld.CreateInBoundsGEP(
          Gbl, {llvm::Constant::getNullValue(CGM.Int64Ty), WarpId});
      Address MediumPtr(MediumPtrVal, C.getTypeAlignInChars(Ty));
      // Casting to actual data type.
      // MediumPtr = (type[i]*)MediumPtrAddr;
      MediumPtr =
          Bld.CreateElementBitCast(MediumPtr, CGF.ConvertTypeForMem(Ty));

      //*MediumPtr = elem
      Bld.CreateStore(Elem, MediumPtr);

      Bld.CreateBr(MergeBB);

      CGF.EmitBlock(ElseBB);
      Bld.CreateBr(MergeBB);
      CGF.EmitBlock(MergeBB);

      // num_thread_to_synchronize = warpNumNeeded * 32
      auto *NumThreadActive =
          Bld.CreateNSWMul(WarpNumVal, Bld.getInt32(DS_Max_Worker_Warp_Size),
                           "num_thread_to_synchronize");
      // named_barrier_sync(num_thread_to_synchronize)
      GetNVPTXBarrier(CGF, PARALLEL_BARRIER, NumThreadActive);

      llvm::BasicBlock *ThenBB2 = CGF.createBasicBlock("then");
      llvm::BasicBlock *ElseBB2 = CGF.createBasicBlock("else");
      llvm::BasicBlock *MergeBB2 = CGF.createBasicBlock("ifcont");

      // if threadId < WarpNeeded
      auto IsActiveThread = Bld.CreateICmpULT(ThreadId, WarpNumVal, "ifcond");
      Bld.CreateCondBr(IsActiveThread, ThenBB2, ElseBB2);

      CGF.EmitBlock(ThenBB2);

      // SrcMediumPtr = &shared[tid]
      llvm::Value *SrcMediumPtrVal = Bld.CreateInBoundsGEP(
          Gbl, {llvm::Constant::getNullValue(CGM.Int64Ty), ThreadId});
      Address SrcMediumPtr(SrcMediumPtrVal, C.getTypeAlignInChars(Ty));
      // SrcMediumVal = *SrcMediumPtr;
      SrcMediumPtr =
          Bld.CreateElementBitCast(SrcMediumPtr, CGF.ConvertTypeForMem(Ty));
      llvm::Value *SrcMediumValue =
          CGF.EmitLoadOfScalar(SrcMediumPtr, /*Volatile=*/false, Ty, Loc);

      //*TargetElemPtr = SrcMediumVal;
      StoreScalarTarget(SrcMediumValue);

      Bld.CreateBr(MergeBB2);

      CGF.EmitBlock(ElseBB2);
      Bld.CreateBr(MergeBB2);

      CGF.EmitBlock(MergeBB2);

      // While master warp copies values from shared memory, all other warps
      // must wait.
      GetNVPTXBarrier(CGF, PARALLEL_BARRIER, NumThreadActive);
    };
    auto &&ScalarCopyMaker = [&CGF, &CopyThruSharedMemory,
                              Loc](QualType Ty, Address Src, Address Dest) {
      auto &&LoadSrc = [&CGF, &Ty, &Src, Loc]() -> llvm::Value * {
        // elem = *elemptr
        llvm::Value *Elem =
            CGF.EmitLoadOfScalar(Src, /*Volatile=*/false, Ty, Loc);
        return Elem;
      };
      auto &&StoreTarget = [&CGF, &Ty, &Dest](llvm::Value *Val) {
        //*TargetElemPtr = SrcMediumVal;
        CGF.EmitStoreOfScalar(Val, Dest, /*Volatile=*/false, Ty);
      };
      CopyThruSharedMemory(Ty, LoadSrc, StoreTarget);
    };
    auto &&ArrayElementCopyMaker = [&CGF, &C, &Bld, &CopyThruSharedMemory, Loc](
        unsigned ElementSize, llvm::Value *Src, llvm::Value *Dest,
        llvm::Value *Idx) {
      auto ElemTy = C.getIntTypeForBitwidth(/*DestWidth*/ ElementSize,
                                            /*Signed*/ false);
      auto &&LoadSrc = [&CGF, &C, &Bld, &ElemTy, &Src, Idx,
                        Loc]() -> llvm::Value * {
        Address ElemPtr(Bld.CreateGEP(Src, Idx), C.getTypeAlignInChars(ElemTy));
        llvm::Value *Elem =
            CGF.EmitLoadOfScalar(ElemPtr, /*Volatile=*/false, ElemTy, Loc);
        return Elem;
      };
      auto &&StoreTarget = [&CGF, &C, &Bld, &ElemTy, &Dest,
                            Idx](llvm::Value *Val) {
        Address ElemPtr(Bld.CreateGEP(Dest, Idx),
                        C.getTypeAlignInChars(ElemTy));
        CGF.EmitStoreOfScalar(Val, ElemPtr, /*Volatile=*/false, ElemTy);
      };
      CopyThruSharedMemory(ElemTy, LoadSrc, StoreTarget);
    };
    CopyReductionVariable(CGF, Private->getType(), ElemPtr, TargetElemPtr,
                          ScalarCopyMaker, ArrayElementCopyMaker,
                          Private->getExprLoc());
    Idx++;
  }

  CGF.FinishFunction();
  return Fn;
}

static llvm::Value *EmitShuffleAndReduceFunction(
    CodeGenModule &CGM, ArrayRef<const Expr *> Privates,
    QualType ReductionArrayTy, llvm::Value *ReduceFn, SourceLocation Loc) {
  //
  // 0.  Algorithm Versions
  // 0.1 Full Warp Reduce (0):
  //     This algorithm assumes that all 32 lanes are active and gathers
  //     data from these 32 lanes, producing a single resultant value.
  // 0.2 Contiguous Partial Warp Reduce (1):
  //     This algorithm assumes that only a *contiguous* subset of lanes
  //     are active. This happens in the last warp when the user specified
  //     thread number is not a integer multiple of 32. In practice, this
  //     contiguous subset always starts with the 0th lane and therefore
  //     the algorithm EXCLUDE the consideration of contiguous live lanes
  //     starting from non-0th lane based on practical consideration.
  // 0.3 Partial Warp Reduce (2):
  //     This algorithm gathers data from any number of lanes at any position.
  // 0.4 Overhead increases as algorithm version increases. The set of problems
  //     every algorithm addresses is a super set of those addressable by
  //     algorithms with a lower version number. All reduced values
  //     are stored in the lowest possible lane.
  //
  // 1.  Terminology
  // 1.1 reduce element:
  //     Reduce element refers to the individual data field with primitive
  //     data types to be combined and reduced across threads.
  // 1.2 reduce_data:
  //     Reduce_data specifically refers to a collection of local,
  //     thread-private
  //     reduce elements.
  // 1.3 remote_reduce_data:
  //     remote_reduce_data refers to a collection of remote (relative to
  //     the current thread) reduce elements.
  // 1.4 alive:
  //     We proceed to distinguish between 3 states of threads that bear
  //     particular importance to the discussion to the implementation
  //     of this ShuffleAndReduce function. We take the most generic
  //     definition for the adjective "alive". It basically means that
  //     the thread is not turned off due to other threads in the same warp
  //     performing a divergent branch conditional.
  // 1.4 active:
  //     Active threads refer to the minimal set of threads that has to be
  //     alive upon entry to this function. The computation is correct iff
  //     active threads are alive. Some threads are alive but they are
  //     not active because they do not contribute to the computation in
  //     any useful manner and turning them off may introduce control
  //     flow overheads without any tangible benefits.
  // 1.5 effective:
  //     In order to comply with the argument requirements of shuffle function,
  //     we have to keep all lanes holding unused data alive. But at most
  //     half of them will be performing value aggregation; we refer to
  //     this half of threads as effective. The other half is simply
  //     handing off data.
  //
  // 2.  Procedure
  // 2.1 step 1 - value shuffle: this step involves all active threads
  //     transfering data from higher lane positions to lower lane
  //     positions, creating remote_reduce_data.
  // 2.2 step 2 - value aggregation: in this step, effective threads
  //     will combine its thread local reduce_data with the
  //     remote_reduce_data and store the resultant value in reduce_data.
  // 2.3 step 3 - value copy: in this step, we deal with the assumption
  //     made by algorithm 2 (i.e. contiguity assumption). When we have
  //     odd number of lanes active, say 2k+1. Only k threads will be
  //     effective and therefore k new values will be produced. However,
  //     reduce_data owned by the (2k+1)th thread is unused, yet it is
  //     k lanes apart from the lane holding the previous unused reduce_data.
  //     Therefore we copy the reduce_data from the (2k+1)th lane to
  //     (k+1)th lane so that the contiguity assumption still holds.
  //
  auto &C = CGM.getContext();

  // Thread local reduce_data used to host the values of data
  // to be reduced.
  ImplicitParamDecl ReduceDataArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                  C.VoidPtrTy, ImplicitParamDecl::Other);
  // Current lane id, could be logical.
  ImplicitParamDecl LaneIDArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.ShortTy,
                              ImplicitParamDecl::Other);
  // Offset of the remote source lane relative to the current lane.
  ImplicitParamDecl OffsetArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.ShortTy,
                              ImplicitParamDecl::Other);
  ImplicitParamDecl AlgoVerArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                               C.ShortTy, ImplicitParamDecl::Other);

  FunctionArgList Args;
  Args.push_back(&ReduceDataArg);
  Args.push_back(&LaneIDArg);
  Args.push_back(&OffsetArg);
  Args.push_back(&AlgoVerArg);
  auto &CGFI = CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      ".omp.reduction.shuffle_and_reduce_func", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, Fn, CGFI);
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, CGFI, Args, Loc, Loc);

  auto &Bld = CGF.Builder;

  Address AddrReduceDataArg = CGF.GetAddrOfLocalVar(&ReduceDataArg);
  Address LocalReduceData(
      Bld.CreatePointerBitCastOrAddrSpaceCast(
          CGF.EmitLoadOfScalar(AddrReduceDataArg, /*Volatile=*/false,
                               C.VoidPtrTy, Loc),
          CGF.ConvertTypeForMem(ReductionArrayTy)->getPointerTo()),
      CGF.getPointerAlign());

  Address AddrOffsetArg = CGF.GetAddrOfLocalVar(&OffsetArg);
  llvm::Value *OffsetArgVal =
      CGF.EmitLoadOfScalar(AddrOffsetArg, /*Volatile=*/false, C.ShortTy, Loc);

  Address AddrLaneIDArg = CGF.GetAddrOfLocalVar(&LaneIDArg);
  llvm::Value *LaneIDArgVal =
      CGF.EmitLoadOfScalar(AddrLaneIDArg, /*Volatile=*/false, C.ShortTy, Loc);

  Address AddrAlgoVerArg = CGF.GetAddrOfLocalVar(&AlgoVerArg);
  llvm::Value *AlgoVerArgVal =
      CGF.EmitLoadOfScalar(AddrAlgoVerArg, /*Volatile=*/false, C.ShortTy, Loc);

  // void* RemoteReduceData[]
  // Create a local thread-private variable to host the reduce_data
  // from a remote lane.
  Address RemoteReduceData =
      CGF.CreateMemTemp(ReductionArrayTy, ".omp.reduction.remote_reduce_list");

  // This loop iterates through the list of reduce elements and copies element
  // by element from a remote lane to the local variable remote_red_list.
  EmitDirectionSpecializedReduceDataCopy(
      Shuffle_To_ReduceData, CGF, ReductionArrayTy, Privates, LocalReduceData,
      RemoteReduceData, OffsetArgVal);

  // The actions to be performed on the remote reduce data is dependent
  // upon the algorithm version.
  //
  //  if (AlgoVer==0) || (AlgoVer==1 && (LaneId < Offset)) || (AlgoVer==2 &&
  //  LaneId % 2 == 0 && Offset > 0)
  //    do the reduction value aggregation (step 2)
  //
  //  the thread local variable reduce_data results are mutated in place
  //  to host the reduced data which is the combined value produced from local
  //  and
  //  remote reduce_data
  //
  //  When AlgoVer==0, the first conjunction evaluates to true, making
  //    the entire predicate true during compile time.
  //  When AlgoVer==1, the second conjunction has only the second part to be
  //    evaluated during runtime. Other conjunctions evaluates to false
  //    during compile time.
  //  When AlgoVer==2, the third conjunction has only the second part to be
  //    evaluated during runtime. Other conjunctions evaluates to false
  //    during compile time.
  //

  auto CondAlgo0 = Bld.CreateICmpEQ(AlgoVerArgVal, Bld.getInt16(0));

  auto CondAlgo1 =
      Bld.CreateAnd(Bld.CreateICmpEQ(AlgoVerArgVal, Bld.getInt16(1)),
                    Bld.CreateICmpULT(LaneIDArgVal, OffsetArgVal));

  auto CondAlgo2 = Bld.CreateAnd(
      Bld.CreateICmpEQ(AlgoVerArgVal, Bld.getInt16(2)),
      Bld.CreateICmpEQ(Bld.CreateAnd(LaneIDArgVal, Bld.getInt16(1)),
                       Bld.getInt16(0)));
  CondAlgo2 = Bld.CreateAnd(CondAlgo2,
                            Bld.CreateICmpSGT(OffsetArgVal, Bld.getInt16(0)));

  auto CondReduce = Bld.CreateOr(CondAlgo0, CondAlgo1);
  CondReduce = Bld.CreateOr(CondReduce, CondAlgo2);

  llvm::BasicBlock *ThenBB = CGF.createBasicBlock("then");
  llvm::BasicBlock *ElseBB = CGF.createBasicBlock("else");
  llvm::BasicBlock *MergeBB = CGF.createBasicBlock("ifcont");
  Bld.CreateCondBr(CondReduce, ThenBB, ElseBB);

  // Then call reduce, results are stored in localData.
  CGF.EmitBlock(ThenBB);

  // reduce_function(LocalReduceData, RemoteReduceData)
  llvm::Value *LocalDataPtr = Bld.CreatePointerBitCastOrAddrSpaceCast(
      LocalReduceData.getPointer(), CGF.VoidPtrTy);
  llvm::Value *RemoteDataPtr = Bld.CreatePointerBitCastOrAddrSpaceCast(
      RemoteReduceData.getPointer(), CGF.VoidPtrTy);
  llvm::SmallVector<llvm::Value *, 2> FnArgs;
  FnArgs.push_back(LocalDataPtr);
  FnArgs.push_back(RemoteDataPtr);
  CGM.getOpenMPRuntime().emitOutlinedFunctionCall(CGF, Loc, ReduceFn, FnArgs);
  Bld.CreateBr(MergeBB);

  CGF.EmitBlock(ElseBB);
  Bld.CreateBr(MergeBB);

  CGF.EmitBlock(MergeBB);

  // if (AlgoVer==1 && (LaneId >= Offset)) copy remote_reduce_data to local
  // reduce_data
  auto CondCopy =
      Bld.CreateAnd(Bld.CreateICmpEQ(AlgoVerArgVal, Bld.getInt16(1)),
                    Bld.CreateICmpUGE(LaneIDArgVal, OffsetArgVal));

  llvm::BasicBlock *CpyThenBB = CGF.createBasicBlock("then");
  llvm::BasicBlock *CpyElseBB = CGF.createBasicBlock("else");
  llvm::BasicBlock *CpyMergeBB = CGF.createBasicBlock("ifcont");
  Bld.CreateCondBr(CondCopy, CpyThenBB, CpyElseBB);

  CGF.EmitBlock(CpyThenBB);
  EmitDirectionSpecializedReduceDataCopy(ReduceData_To_ReduceData, CGF,
                                         ReductionArrayTy, Privates,
                                         RemoteReduceData, LocalReduceData);
  Bld.CreateBr(CpyMergeBB);

  CGF.EmitBlock(CpyElseBB);
  Bld.CreateBr(CpyMergeBB);

  CGF.EmitBlock(CpyMergeBB);

  CGF.FinishFunction();
  return Fn;
}

void CGOpenMPRuntimeNVPTX::emitReduction(
    CodeGenFunction &CGF, SourceLocation Loc, ArrayRef<const Expr *> Privates,
    ArrayRef<const Expr *> LHSExprs, ArrayRef<const Expr *> RHSExprs,
    ArrayRef<const Expr *> ReductionOps, bool WithNowait, bool SimpleReduction,
    OpenMPDirectiveKind ReductionKind) {
  if (!CGF.HaveInsertPoint())
    return;

  bool TeamsReduction = isOpenMPTeamsDirective(ReductionKind);
  bool ParallelReduction = isOpenMPParallelDirective(ReductionKind);
  bool SimdReduction = isOpenMPSimdDirective(ReductionKind);
  assert((TeamsReduction || ParallelReduction || SimdReduction) &&
         "Invalid reduction selection in emitReduction.");

  auto &C = CGM.getContext();

  // 1. Build a list of reduction variables.
  // void *RedList[<n>] = {<ReductionVars>[0], ..., <ReductionVars>[<n>-1]};
  auto Size = RHSExprs.size();
  for (auto *E : Privates) {
    if (E->getType()->isVariablyModifiedType())
      // Reserve place for array size.
      ++Size;
  }
  llvm::APInt ArraySize(/*unsigned int numBits=*/32, Size);
  QualType ReductionArrayTy =
      C.getConstantArrayType(C.VoidPtrTy, ArraySize, ArrayType::Normal,
                             /*IndexTypeQuals=*/0);
  Address ReductionList =
      CGF.CreateMemTemp(ReductionArrayTy, ".omp.reduction.red_list");
  auto IPriv = Privates.begin();
  unsigned Idx = 0;
  for (unsigned I = 0, E = RHSExprs.size(); I < E; ++I, ++IPriv, ++Idx) {
    Address Elem = CGF.Builder.CreateConstArrayGEP(ReductionList, Idx,
                                                   CGF.getPointerSize());
    CGF.Builder.CreateStore(
        CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
            CGF.EmitLValue(RHSExprs[I]).getPointer(), CGF.VoidPtrTy),
        Elem);
    if ((*IPriv)->getType()->isVariablyModifiedType()) {
      // Store array size.
      ++Idx;
      Elem = CGF.Builder.CreateConstArrayGEP(ReductionList, Idx,
                                             CGF.getPointerSize());
      llvm::Value *Size = CGF.Builder.CreateIntCast(
          CGF.getVLASize(
                 CGF.getContext().getAsVariableArrayType((*IPriv)->getType()))
              .first,
          CGF.SizeTy, /*isSigned=*/false);
      CGF.Builder.CreateStore(CGF.Builder.CreateIntToPtr(Size, CGF.VoidPtrTy),
                              Elem);
    }
  }

  // 2. Emit reduce_func().
  auto *ReductionFn = emitReductionFunction(
      CGM, Loc, CGF.ConvertTypeForMem(ReductionArrayTy)->getPointerTo(),
      Privates, LHSExprs, RHSExprs, ReductionOps);

  // 4. Build res = __kmpc_reduce{_nowait}(<gtid>, <n>, sizeof(RedList),
  // RedList, shuffle_reduce_func, interwarp_copy_func);
  auto *ThreadId = getThreadID(CGF, Loc);
  auto *ReductionArrayTySize = CGF.getTypeSize(ReductionArrayTy);
  auto *RL = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      ReductionList.getPointer(), CGF.VoidPtrTy);

  auto *ShuffleAndReduce = EmitShuffleAndReduceFunction(
      CGM, Privates, ReductionArrayTy, ReductionFn, Loc);
  auto *InterWarpCopy =
      EmitInterWarpCopyFunction(CGM, Privates, ReductionArrayTy, Loc);
  auto *ScratchPadCopy =
      EmitCopyToScratchpad(CGM, Privates, ReductionArrayTy, Loc);
  auto *LoadAndReduce = EmitReduceScratchpadFunction(
      CGM, Privates, ReductionArrayTy, ReductionFn, Loc);
  llvm::Value *Args[] = {
      ThreadId,                              // i32 <gtid>
      CGF.Builder.getInt32(RHSExprs.size()), // i32 <n>
      ReductionArrayTySize,                  // size_type sizeof(RedList)
      RL,                                    // void *RedList
      ShuffleAndReduce, // void (*kmp_ShuffleReductFctPtr)(void *rhsData,
                        // int16_t lane_id, int16_t lane_offset, int16_t
                        // shortCircuit);
      InterWarpCopy     // void (*kmp_InterWarpCopyFctPtr)(void* src, int
                        // warp_num);
  };

  llvm::CallInst *Res = nullptr;
  if (ParallelReduction) {
    Res = CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(selectRuntimeCall<OpenMPRTLFunctionNVPTX>(
            isSPMDExecutionMode(), isOMPRuntimeInitialized(),
            {OMPRTL_NVPTX__kmpc_parallel_reduce_nowait_simple_spmd,
             OMPRTL_NVPTX__kmpc_parallel_reduce_nowait_simple_generic,
             OMPRTL_NVPTX__kmpc_parallel_reduce_nowait})),
        Args);
  } else if (SimdReduction) {
    Res = CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(
            /*WithNowait ? */ OMPRTL_NVPTX__kmpc_simd_reduce_nowait
            /*,OMPRTL__kmpc_reduce*/),
        Args);
  }

  // ReductionKind of OMPD_distribute_parallel_for is used to indicate
  // reduction codegen of coalesced 'distribute parallel for' in a combined
  // directive such as 'target teams distribute parallel for'. In this case
  // return early since this function will be called again on the teams
  // reduction directive.
  if (ReductionKind == OMPD_distribute_parallel_for)
    return;

  if (TeamsReduction) {
    llvm::Value *TeamsArgs[] = {
        ThreadId,                              // i32 <gtid>
        CGF.Builder.getInt32(RHSExprs.size()), // i32 <n>
        ReductionArrayTySize,                  // size_type sizeof(RedList)
        RL,                                    // void *RedList
        ShuffleAndReduce, // void (*kmp_ShuffleReductFctPtr)(void *rhsData,
                          // int16_t lane_id, int16_t lane_offset, int16_t
                          // shortCircuit);
        InterWarpCopy,    // void (*kmp_InterWarpCopyFctPtr)(void* src, int
                          // warp_num);
        ScratchPadCopy,   // (*kmp_CopyToScratchpadFctPtr)(void *reduceData,
                          // void * scratchpad, int32_t index, int32_t width);
        LoadAndReduce     // (*kmp_LoadReduceFctPtr)(void *reduceData,
                          // void * scratchpad, int32_t index,
                          // int32_t width, int32_t reduce);
    };
    Res = CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(selectRuntimeCall<OpenMPRTLFunctionNVPTX>(
            isSPMDExecutionMode(), isOMPRuntimeInitialized(),
            {OMPRTL_NVPTX__kmpc_teams_reduce_nowait_simple_spmd,
             OMPRTL_NVPTX__kmpc_teams_reduce_nowait_simple_generic,
             OMPRTL_NVPTX__kmpc_teams_reduce_nowait})),
        TeamsArgs);
  }

  // 5. Build switch(res)
  auto *DefaultBB = CGF.createBasicBlock(".omp.reduction.default");
  auto *SwInst = CGF.Builder.CreateSwitch(Res, DefaultBB, /*NumCases=*/1);

  // 6. Build case 1: where we have reduced values in the master
  //    thread in each team.
  //    Team reduction implementation is pending
  //    __kmpc_end_reduce{_nowait}(<gtid>);
  //    break;
  auto *Case1BB = CGF.createBasicBlock(".omp.reduction.case1");
  SwInst->addCase(CGF.Builder.getInt32(1), Case1BB);
  CGF.EmitBlock(Case1BB);

  // Add emission of __kmpc_end_reduce{_nowait}(<gtid>);
  llvm::Value *EndArgs[] = {
      ThreadId // i32 <gtid>
  };
  auto &&CodeGen = [&Privates, &LHSExprs, &RHSExprs, &ReductionOps,
                    this](CodeGenFunction &CGF, PrePostActionTy &Action) {
    auto IPriv = Privates.begin();
    auto ILHS = LHSExprs.begin();
    auto IRHS = RHSExprs.begin();
    for (auto *E : ReductionOps) {
      emitSingleReductionCombiner(CGF, E, *IPriv, cast<DeclRefExpr>(*ILHS),
                                  cast<DeclRefExpr>(*IRHS));
      ++IPriv;
      ++ILHS;
      ++IRHS;
    }
  };
  RegionCodeGenTy RCG(CodeGen);
  CommonActionTy Action(
      nullptr, llvm::None,
      createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_end_reduce_nowait),
      EndArgs);
  RCG.setAction(Action);
  RCG(CGF);
  CGF.EmitBranch(DefaultBB);
  CGF.EmitBlock(DefaultBB, /*IsFinished=*/true);
}

const VarDecl *
CGOpenMPRuntimeNVPTX::translateParameter(const FieldDecl *FD,
                                         const VarDecl *NativeParam) const {
  if (!NativeParam->getType()->isReferenceType())
    return NativeParam;
  QualType ArgType = NativeParam->getType();
  QualifierCollector QC;
  const Type *NonQualTy = QC.strip(ArgType);
  QualType PointeeTy = cast<ReferenceType>(NonQualTy)->getPointeeType();
  if (const auto *Attr = FD->getAttr<OMPCaptureKindAttr>()) {
    if (Attr->getCaptureKind() == OMPC_map) {
      PointeeTy = CGM.getContext().getAddrSpaceQualType(PointeeTy,
                                                        LangAS::opencl_global);
    }
  }
  ArgType = CGM.getContext().getPointerType(PointeeTy);
  QC.addRestrict();
  enum { NVPTX_local_addr = 5 };
  QC.addAddressSpace(NVPTX_local_addr);
  ArgType = QC.apply(CGM.getContext(), ArgType);
  if (isa<ImplicitParamDecl>(NativeParam)) {
    return ImplicitParamDecl::Create(
        CGM.getContext(), /*DC=*/nullptr, NativeParam->getLocation(),
        NativeParam->getIdentifier(), ArgType, ImplicitParamDecl::Other);
  }
  return ParmVarDecl::Create(
      CGM.getContext(),
      const_cast<DeclContext *>(NativeParam->getDeclContext()),
      NativeParam->getLocStart(), NativeParam->getLocation(),
      NativeParam->getIdentifier(), ArgType,
      /*TInfo=*/nullptr, SC_None, /*DefArg=*/nullptr);
}

Address
CGOpenMPRuntimeNVPTX::getParameterAddress(CodeGenFunction &CGF,
                                          const VarDecl *NativeParam,
                                          const VarDecl *TargetParam) const {
  assert(NativeParam != TargetParam &&
         NativeParam->getType()->isReferenceType() &&
         "Native arg must not be the same as target arg.");
  Address LocalAddr = CGF.GetAddrOfLocalVar(TargetParam);
  QualType NativeParamType = NativeParam->getType();
  QualifierCollector QC;
  const Type *NonQualTy = QC.strip(NativeParamType);
  QualType NativePointeeTy = cast<ReferenceType>(NonQualTy)->getPointeeType();
  unsigned NativePointeeAddrSpace =
      NativePointeeTy.getQualifiers().getAddressSpace();
  QualType TargetTy = TargetParam->getType();
  llvm::Value *TargetAddr = CGF.EmitLoadOfScalar(
      LocalAddr, /*Volatile=*/false, TargetTy, TargetParam->getLocation());
  // First cast to generic.
  TargetAddr = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      TargetAddr, TargetAddr->getType()->getPointerElementType()->getPointerTo(
                      /*AddrSpace=*/0));
  // Cast from generic to native address space.
  TargetAddr = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      TargetAddr, TargetAddr->getType()->getPointerElementType()->getPointerTo(
                      NativePointeeAddrSpace));
  Address NativeParamAddr = CGF.CreateMemTemp(NativeParamType);
  CGF.EmitStoreOfScalar(TargetAddr, NativeParamAddr, /*Volatile=*/false,
                        NativeParamType);
  return NativeParamAddr;
}

void CGOpenMPRuntimeNVPTX::emitOutlinedFunctionCall(
    CodeGenFunction &CGF, SourceLocation Loc, llvm::Value *OutlinedFn,
    ArrayRef<llvm::Value *> Args) const {
  SmallVector<llvm::Value *, 4> TargetArgs;
  TargetArgs.reserve(Args.size());
  auto *FnType =
      cast<llvm::FunctionType>(OutlinedFn->getType()->getPointerElementType());
  for (unsigned I = 0, E = Args.size(); I < E; ++I) {
    if (FnType->isVarArg() && FnType->getNumParams() <= I) {
      TargetArgs.append(std::next(Args.begin(), I), Args.end());
      break;
    }
    llvm::Type *TargetType = FnType->getParamType(I);
    llvm::Value *NativeArg = Args[I];
    if (!TargetType->isPointerTy()) {
      TargetArgs.emplace_back(NativeArg);
      continue;
    }
    llvm::Value *TargetArg = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
        NativeArg, NativeArg->getType()->getPointerElementType()->getPointerTo(
                       /*AddrSpace=*/0));
    TargetArgs.emplace_back(
        CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(TargetArg, TargetType));
  }
  CGOpenMPRuntime::emitOutlinedFunctionCall(CGF, Loc, OutlinedFn, TargetArgs);
}
