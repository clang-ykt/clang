//===----- CGOpenMPRuntimeNVPTX.h - Interface to OpenMP NVPTX Runtimes ----===//
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

#ifndef LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMENVPTX_H
#define LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMENVPTX_H

#include "CGOpenMPRuntime.h"
#include "CodeGenFunction.h"
#include "clang/AST/StmtOpenMP.h"
#include "llvm/IR/CallSite.h"

namespace clang {
namespace CodeGen {

class CGOpenMPRuntimeNVPTX : public CGOpenMPRuntime {
  //
  // NVPTX calls.
  //

  /// \brief Get the GPU warp size.
  llvm::Value *getNVPTXWarpSize(CodeGenFunction &CGF);

  /// \brief Get the id of the current thread on the GPU.
  llvm::Value *getNVPTXThreadID(CodeGenFunction &CGF);

  /// \brief Get the id of the current block on the GPU.
  llvm::Value *getNVPTXBlockID(CodeGenFunction &CGF);

  // \brief Get the maximum number of threads in a block of the GPU.
  llvm::Value *getNVPTXNumThreads(CodeGenFunction &CGF);

  /// \brief Get barrier to synchronize all threads in a block.
  void getNVPTXCTABarrier(CodeGenFunction &CGF);

  /// \brief Get barrier #n to synchronize selected (multiple of 32) threads in
  /// a block.
  void getNVPTXBarrier(CodeGenFunction &CGF, int ID, int NumThreads);

  // \brief Synchronize all GPU threads in a block.
  void syncCTAThreads(CodeGenFunction &CGF);

  //
  // OMP calls.
  //

  /// \brief Get the thread id of the OMP master thread.
  /// The master thread id is the first thread (lane) of the last warp in the
  /// GPU block.  Warp size is assumed to be some power of 2.
  /// Thread id is 0 indexed.
  /// E.g: If NumThreads is 33, master id is 32.
  ///      If NumThreads is 64, master id is 32.
  ///      If NumThreads is 1024, master id is 992.
  llvm::Value *getMasterThreadID(CodeGenFunction &CGF);

  /// \brief Get number of OMP workers for parallel region after subtracting
  /// the master warp.
  llvm::Value *getNumWorkers(CodeGenFunction &CGF);

  /// \brief Get thread id in team.
  /// FIXME: Remove the expensive remainder operation.
  llvm::Value *getTeamThreadId(CodeGenFunction &CGF);

  /// \brief Get global thread id.
  llvm::Value *getGlobalThreadId(CodeGenFunction &CGF);

  //
  // Private state and methods.
  //

  // Pointers to outlined function work for workers.
  llvm::SmallVector<llvm::Function *, 16> Work;

  class EntryFunctionState {
  public:
    llvm::BasicBlock *ExitBB;

    EntryFunctionState() : ExitBB(nullptr){};
  };

  class WorkerFunctionState {
  public:
    llvm::Function *WorkerFn;
    const CGFunctionInfo *CGFI;

    WorkerFunctionState(CodeGenModule &CGM);

  private:
    void createWorkerFunction(CodeGenModule &CGM);
  };

  /// \brief Emit the worker function for the current target region.
  void emitWorkerFunction(WorkerFunctionState &WST);

  /// \brief Helper for worker function. Emit body of worker loop.
  void emitWorkerLoop(CodeGenFunction &CGF, WorkerFunctionState &WST);

  /// \brief Helper for target entry function. Guide the master and worker
  /// threads to their respective locations.
  void emitEntryHeader(CodeGenFunction &CGF, EntryFunctionState &EST,
                       WorkerFunctionState &WST);

  /// \brief Signal termination of OMP execution.
  void emitEntryFooter(CodeGenFunction &CGF, EntryFunctionState &EST);

  /// \brief Returns specified OpenMP runtime function for the current OpenMP
  /// implementation.  Specialized for the NVPTX device.
  /// \param Function OpenMP runtime function.
  /// \return Specified function.
  llvm::Constant *createNVPTXRuntimeFunction(unsigned Function);

  /// \brief Gets thread id value for the current thread.
  ///
  llvm::Value *getThreadID(CodeGenFunction &CGF, SourceLocation Loc) override;

  /// \brief Emits captured variables for the outlined function for the
  /// specified OpenMP parallel directive \a D.
  void
  emitCapturedVars(CodeGenFunction &CGF, const OMPExecutableDirective &S,
                   llvm::SmallVector<llvm::Value *, 16> &CapturedVars) override;

  //
  // Base class overrides.
  //

  /// \brief Creates offloading entry for the provided entry ID \a ID,
  /// address \a Addr and size \a Size.
  void createOffloadEntry(llvm::Constant *ID, llvm::Constant *Addr,
                          uint64_t Size) override;

  /// \brief Emit outlined function for 'target' directive on the NVPTX
  /// device.
  /// \param D Directive to emit.
  /// \param ParentName Name of the function that encloses the target region.
  /// \param OutlinedFn Outlined function value to be defined by this call.
  /// \param OutlinedFnID Outlined function ID value to be defined by this call.
  /// \param IsOffloadEntry True if the outlined function is an offload entry.
  /// An outlined function may not be an entry if, e.g. the if clause always
  /// evaluates to false.
  void emitTargetOutlinedFunction(const OMPExecutableDirective &D,
                                  StringRef ParentName,
                                  llvm::Function *&OutlinedFn,
                                  llvm::Constant *&OutlinedFnID,
                                  bool IsOffloadEntry) override;

public:
  explicit CGOpenMPRuntimeNVPTX(CodeGenModule &CGM);

  /// \brief Emits code for parallel or serial call of the \a OutlinedFn with
  /// variables captured in a record which address is stored in \a
  /// CapturedStruct.
  /// \param OutlinedFn Outlined function to be run in parallel threads. Type of
  /// this function is void(*)(kmp_int32 *, kmp_int32, struct context_vars*).
  /// \param CapturedVars A pointer to the record with the references to
  /// variables used in \a OutlinedFn function.
  /// \param IfCond Condition in the associated 'if' clause, if it was
  /// specified, nullptr otherwise.
  ///
  void emitParallelCall(CodeGenFunction &CGF, SourceLocation Loc,
                        llvm::Value *OutlinedFn,
                        ArrayRef<llvm::Value *> CapturedVars,
                        const Expr *IfCond) override;

  /// \brief Emits outlined function for the specified OpenMP parallel directive
  /// \a D. This outlined function has type void(*)(kmp_int32 *ThreadID,
  /// kmp_int32 BoundID, struct context_vars*).
  /// \param D OpenMP directive.
  /// \param ThreadIDVar Variable for thread id in the current OpenMP region.
  /// \param InnermostKind Kind of innermost directive (for simple directives it
  /// is a directive itself, for combined - its innermost directive).
  /// \param CodeGen Code generation sequence for the \a D directive.
  llvm::Value *
  emitParallelOrTeamsOutlinedFunction(const OMPExecutableDirective &D,
                               const VarDecl *ThreadIDVar,
                               OpenMPDirectiveKind InnermostKind,
                               const RegionCodeGenTy &CodeGen) override;

  /// \brief Check if we should generate code as if \a ScheduleKind is static
  /// with a chunk size of 1.
  /// \param ScheduleKind Schedule Kind specified in the 'schedule' clause.
  /// \param Chunk size.
  ///
  bool generateCoalescedSchedule(OpenMPScheduleClauseKind ScheduleKind,
                                 bool ChunkSizeOne,
                                 bool ordered) const override;

  /// \brief Check if we must always generate a barrier at the end of a
  /// particular construct regardless of the presence of a nowait clause.
  /// This may occur when a particular offload device does not support
  /// concurrent execution of certain directive and clause combinations.
  bool requiresBarrier(const OMPLoopDirective &S) const override;
};

} // CodeGen namespace.
} // clang namespace.

#endif // LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMENVPTX_H
