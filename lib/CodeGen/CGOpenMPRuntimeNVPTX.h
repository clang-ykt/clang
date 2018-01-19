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
  llvm::StringMap<StringRef> stdFuncs;
  StringRef RenameStandardFunction(StringRef name) override;

  ///
  /// \param CGF Reference to current CodeGenFunction.
  /// \param Loc Clang source location.
  /// \param SchedKind Schedule kind, specified by the 'dist_schedule' clause.
  /// \param Values Input arguments for the construct.
  /// \param CoalescedDistSchedule Indicates if coalesced scheduling type is
  /// required.
  ///
  virtual void
  emitDistributeStaticInit(CodeGenFunction &CGF, SourceLocation Loc,
                           OpenMPDistScheduleClauseKind SchedKind,
                           const StaticRTInput &Values,
                           bool CoalescedDistSchedule = false) override;

  /// \brief Call the appropriate runtime routine to notify that we finished
  /// all the work with current loop.
  ///
  /// \param CGF Reference to current CodeGenFunction.
  /// \param Loc Clang source location.
  /// \param DKind Kind of the directive for which the static finish is emitted.
  /// \param CoalescedDistSchedule Indicates if coalesced scheduling type is
  /// required.
  ///
  virtual void emitForStaticFinish(CodeGenFunction &CGF, SourceLocation Loc,
                                   OpenMPDirectiveKind DKind,
                                   bool CoalescedDistSchedule = false) override;

  //
  // Data Sharing related calls.
  //

  // \brief Type of the data sharing master slot. By default the size is zero
  // meaning that the data size is to be determined.
  QualType DataSharingMasterSlotQty;
  QualType getDataSharingMasterSlotQty();

  // \brief Type of the data sharing worker warp slot. By default the size is
  // zero meaning that the data size is to be determined.
  QualType DataSharingWorkerWarpSlotQty;
  QualType getDataSharingWorkerWarpSlotQty();

  // \brief Get the type of the master or worker slot incomplete.
  QualType DataSharingSlotQty;
  QualType getDataSharingSlotQty(bool UseFixedDataSize = false,
                                 bool IsMaster = false);
  llvm::Type *getDataSharingSlotTy(bool UseFixedDataSize = false,
                                   bool IsMaster = false);

  // \brief Type of the data sharing root slot.
  QualType DataSharingRootSlotQty;
  QualType getDataSharingRootSlotQty();

  // \brief Return address of the initial slot that is used to share data.
  LValue getDataSharingRootSlotLValue(CodeGenFunction &CGF, bool IsMaster);

  // \brief Initialize the data sharing slots and pointers and return the
  // generated call.
  void initializeDataSharing(CodeGenFunction &CGF, bool IsMaster);

  // \brief Initialize the data sharing slots and pointers and return the
  // generated call.
  llvm::Function *
  createKernelInitializerFunction(llvm::Function *WorkerFunction,
                                  bool RequiresOMPRuntime);

protected:
  /// \brief Returns __kmpc_for_static_init_* runtime function for the specified
  /// size \a IVSize and sign \a IVSigned.
  virtual llvm::Constant *createForStaticInitFunction(unsigned IVSize,
                                                      bool IVSigned) override;

public:
  // \brief Group the captures information for a given context.
  struct DataSharingInfo {
    enum DataSharingType {
      // A value allocated in the current function - the alloca has to be
      // replaced by the address in shared memory.
      DST_Val,
      // A reference captured into this function - the reference has to be
      // shared as is.
      DST_Ref,
      // A value allocated in the current function but required a cast in the
      // header - it has to be replaced by the address in shared memory and the
      // pointee has to be copied there.
      DST_Cast,
    };
    // The local values of the captures. The boolean indicates that what is
    // being shared is a reference and not the variable original storage.
    llvm::SmallVector<std::pair<const VarDecl *, DataSharingType>, 8>
        CapturesValues;
    llvm::SmallVector<std::pair<const Expr*, const VarDecl *>, 8>
        VLADeclMap;

    void add(const VarDecl *VD, DataSharingType DST) {
      CapturesValues.push_back(std::make_pair(VD, DST));
    }

    void addVLADecl(const Expr* VATExpr, const VarDecl *VD) {
      // VLADeclMap[VATExpr] = VD;
      VLADeclMap.push_back(std::make_pair(VATExpr, VD));
    }

    const VarDecl *getVLADecl(const Expr* VATExpr) const {
      for (auto ExprDeclPair : VLADeclMap) {
        if (ExprDeclPair.first == VATExpr) {
          return ExprDeclPair.second;
        }
      }
      assert(false && "No VAT expression that matches the input");
      return nullptr;
    }

    bool isVLADecl(const VarDecl* VD) const {
      for (auto ExprDeclPair : VLADeclMap) {
        if (ExprDeclPair.second == VD) {
          return true;
        }
      }
      return false;
    }

    // The record type of the sharing region if shared by the master.
    QualType MasterRecordType;
    // The record type of the sharing region if shared by the worker warps.
    QualType WorkerWarpRecordType;
  };

  /// \brief Specialization of codegen based on Programming models of the
  /// OpenMP construct.
  enum ExecutionMode {
    /// \brief Single Program Multiple Data.
    SPMD,
    /// \brief Generic codegen to support fork-join model.
    GENERIC,
    UNKNOWN,
  };

  enum MatchReasonCodeKind {
    Unknown,
    ExternFunctionDefinition,
    DirectiveRequiresRuntime,
    NestedParallelRequiresRuntime,
    MasterContextExceedsSharedMemory,
  };

  struct MatchReasonTy {
    MatchReasonCodeKind RC;
    SourceLocation Loc;
    MatchReasonTy(MatchReasonCodeKind RC, SourceLocation Loc)
        : RC(RC), Loc(Loc) {}
    MatchReasonTy() : RC(Unknown), Loc(SourceLocation()) {}
  };

private:
  // \brief Map between a context and its data sharing information.
  typedef llvm::DenseMap<const Decl *, DataSharingInfo> DataSharingInfoMapTy;
  DataSharingInfoMapTy DataSharingInfoMap;

  // \brief Obtain the data sharing info for the current context.
  const DataSharingInfo &getDataSharingInfo(const Decl *Context);

  // \brief Create the data sharing info for the current context.
  void createDataSharingInfo(CodeGenFunction &CGF);

  // \brief Set of all functions that are offload entry points.
  llvm::SmallPtrSet<llvm::Function *, 16> EntryPointFunctionSet;

  // \brief Map between a function and its associated data sharing related
  // values.
  struct DataSharingFunctionInfo {
    bool RequiresOMPRuntime;
    bool IsEntryPoint;
    llvm::Function *EntryWorkerFunction;
    llvm::BasicBlock *EntryExitBlock;
    llvm::BasicBlock *InitDSBlock;
    llvm::Function *InitializationFunction;
    SmallVector<std::pair<llvm::Value *, bool>, 16> ValuesToBeReplaced;
    DataSharingFunctionInfo()
        : RequiresOMPRuntime(true), IsEntryPoint(false),
          EntryWorkerFunction(nullptr), EntryExitBlock(nullptr),
          InitDSBlock(nullptr), InitializationFunction(nullptr) {}
  };
  typedef llvm::DenseMap<llvm::Function *, DataSharingFunctionInfo>
      DataSharingFunctionInfoMapTy;
  DataSharingFunctionInfoMapTy DataSharingFunctionInfoMap;

  // \brief Create the data sharing replacement pairs at the top of a function
  // with parallel regions. If they were created already, do not do anything.
  void
  createDataSharingPerFunctionInfrastructure(CodeGenFunction &EnclosingCGF);

  // \brief Create the data sharing arguments and call the parallel outlined
  // function.
  llvm::Function *createDataSharingParallelWrapper(
      llvm::Function &OutlinedParallelFn, const OMPExecutableDirective &D,
      const Decl *CurrentContext, bool IsSimd = false);

  // \brief Map between an outlined function and its data-sharing-wrap version.
  llvm::DenseMap<llvm::Function *, llvm::Function *> WrapperFunctionsMap;

  // \brief Context that is being currently used for purposes of parallel region
  // code generarion.
  const Decl *CurrentParallelContext = nullptr;

  //
  // NVPTX calls.
  //

  // \brief Get a 32 bit mask, whose bits set to 1 represent the active threads.
  llvm::Value *getNVPTXWarpActiveThreadsMask(CodeGenFunction &CGF);

  // \brief Get the number of active threads in a warp.
  llvm::Value *getNVPTXWarpActiveNumThreads(CodeGenFunction &CGF);

  // \brief Get the ID of the thread among the current active threads in the
  // warp.
  llvm::Value *getNVPTXWarpActiveThreadID(CodeGenFunction &CGF);

  // \brief Get a conditional that is set to true if the thread is the master of
  // the active threads in the warp.
  llvm::Value *getNVPTXIsWarpActiveMaster(CodeGenFunction &CGF);

  //
  // Private state and methods.
  //

  // Pointers to outlined function work for workers.
  llvm::SmallVector<llvm::Function *, 16> Work;

  class TargetKernelProperties {
  public:
    TargetKernelProperties(const CodeGenModule &CGM,
                           const OMPExecutableDirective &D)
        : CGM(CGM), D(D), Mode(CGOpenMPRuntimeNVPTX::ExecutionMode::UNKNOWN),
          RequiresOMPRuntime(true), RequiresDataSharing(true),
          MayContainOrphanedParallel(true),
          HasAtMostOneNestedParallelInLexicalScope(false),
          MasterSharedDataSize(0), ReductionVariableCount(0),
          ReductionSizeInBytes(0) {
      assert(isOpenMPTargetExecutionDirective(D.getDirectiveKind()) &&
             "Expecting a target execution directive.");
      setExecutionMode();
      setRequiresDataSharing();
      setMasterSharedDataSize();
      setRequiresOMPRuntime();
      setMayContainOrphanedParallel();
      setHasAtMostOneNestedParallelInLexicalScope();
      setTeamsReductionInfo();
    };

    CGOpenMPRuntimeNVPTX::ExecutionMode getExecutionMode() const {
      return Mode;
    }

    bool requiresOMPRuntime() const { return RequiresOMPRuntime; }

    MatchReasonTy requiresOMPRuntimeReason() const {
      return RequiresOMPRuntimeReason;
    }

    bool requiresDataSharing() const { return RequiresDataSharing; }

    bool mayContainOrphanedParallel() const {
      return MayContainOrphanedParallel;
    }

    bool hasAtMostOneL1ParallelRegion() const {
      // 'HasAtMostOneNestedParallelInLexicalScope' counts the number of
      // parallel regions in the lexical scope of the target.  If there
      // can be one or more orphaned parallel regions, return false.  Note
      // that this says nothing about L2 parallelism, i.e., a parallel
      // within another parallel region.
      return HasAtMostOneNestedParallelInLexicalScope &&
             !MayContainOrphanedParallel;
    }

    unsigned masterSharedDataSize() const { return MasterSharedDataSize; }

    unsigned getReductionVariableCount() const {
      return ReductionVariableCount;
    }

    unsigned getReductionSizeInBytes() const { return ReductionSizeInBytes; }

  private:
    const CodeGenModule &CGM;
    const OMPExecutableDirective &D;

    // Code generation mode for the target directive.
    CGOpenMPRuntimeNVPTX::ExecutionMode Mode;
    // Record if the target region requires an OpenMP runtime.  For simple
    // kernels it is possible to disable the runtime and thus reduce
    // execution overhead.
    bool RequiresOMPRuntime;
    MatchReasonTy RequiresOMPRuntimeReason;
    // Record if the target region requires data sharing support.  Data
    // sharing support is not required for an SPMD construct if it does not
    // contain a nested parallel or simd directive.
    bool RequiresDataSharing;
    // Record if the target region may encounter an orphaned parallel
    // directive, i.e., a parallel directive in a 'declare target' function.
    bool MayContainOrphanedParallel;
    // Record if the target region has at most a single nested parallel
    // region in its lexical scope.
    bool HasAtMostOneNestedParallelInLexicalScope;
    // Approximate the size in bytes of variables to be shared from master
    // to workers.
    unsigned MasterSharedDataSize;
    // Number of teams reduction variables on the target construct.
    unsigned ReductionVariableCount;
    // Total size of teams reduction variables in bytes.
    unsigned ReductionSizeInBytes;

    void setExecutionMode();

    void setRequiresOMPRuntime();

    // Check if the current target region requires data sharing support.
    // Data sharing support is required if this SPMD construct may have a nested
    // parallel or simd directive.
    void setRequiresDataSharing();

    void setMayContainOrphanedParallel();

    void setHasAtMostOneNestedParallelInLexicalScope();

    void setMasterSharedDataSize();

    void setTeamsReductionInfo();
  };

  class EntryFunctionState {
  public:
    const TargetKernelProperties &TP;
    llvm::BasicBlock *ExitBB;

    EntryFunctionState(CodeGenModule &CGM, const TargetKernelProperties &TP)
        : TP(TP), ExitBB(nullptr){};
  };

  class WorkerFunctionState {
  public:
    const TargetKernelProperties &TP;
    llvm::Function *WorkerFn;
    const CGFunctionInfo *CGFI;
    SourceLocation Loc;

    WorkerFunctionState(CodeGenModule &CGM, const TargetKernelProperties &TP,
                        SourceLocation Loc)
        : TP(TP), WorkerFn(nullptr), CGFI(nullptr), Loc(Loc) {
      createWorkerFunction(CGM);
    };

  private:
    void createWorkerFunction(CodeGenModule &CGM);
  };

  // State information to track orphaned directives.
  bool IsOrphaned;
  // Track parallel nesting level.
  unsigned ParallelNestingLevel;
  // Track whether the OMP runtime is available or elided for the
  // target region.
  bool IsOMPRuntimeInitialized;

  // The current codegen mode.  This is used to customize code generation of
  // certain constructs.
  ExecutionMode CurrMode;

  /// \brief Emit the worker function for the current target region.
  void emitWorkerFunction(WorkerFunctionState &WST);

  /// \brief Helper for worker function. Emit body of worker loop.
  void emitWorkerLoop(CodeGenFunction &CGF, WorkerFunctionState &WST);

  /// \brief Helper for generic target entry function. Guide the master and
  /// worker threads to their respective locations.
  void emitGenericEntryHeader(CodeGenFunction &CGF, EntryFunctionState &EST,
                              WorkerFunctionState &WST);

  /// \brief Signal termination of OMP execution for generic target entry
  /// function.
  void emitGenericEntryFooter(CodeGenFunction &CGF, EntryFunctionState &EST);

  /// \brief Helper for SPMD target entry function.
  void emitSPMDEntryHeader(CodeGenFunction &CGF, EntryFunctionState &EST,
                           const OMPExecutableDirective &D);

  /// \brief Signal termination of SPMD OMP execution.
  void emitSPMDEntryFooter(CodeGenFunction &CGF, EntryFunctionState &EST);

  /// \brief Returns specified OpenMP runtime function for the current OpenMP
  /// implementation.  Specialized for the NVPTX device.
  /// \param Function OpenMP runtime function.
  /// \return Specified function.
  llvm::Constant *createNVPTXRuntimeFunction(unsigned Function);

  /// \brief Gets thread id value for the current thread.
  ///
  llvm::Value *getThreadID(CodeGenFunction &CGF, SourceLocation Loc) override;

  /// Get parallel level of this thread.
  /// Return value is zero (level 0), one (level 1), or two (> level 1).
  llvm::Value *getParallelLevel(CodeGenFunction &CGF, SourceLocation Loc);

  /// \brief Registers the context of a parallel region with the runtime
  /// codegen implementation.
  void registerParallelContext(CodeGenFunction &CGF,
                               const OMPExecutableDirective &S) override;

  //
  // Base class overrides.
  //

  /// \brief Creates offloading entry for the provided entry ID \a ID,
  /// address \a Addr and size \a Size with flags \a Flags.
  void createOffloadEntry(llvm::Constant *ID, llvm::Constant *Addr,
                          uint64_t Size, uint64_t Flags = 0u) override;

  // Create a unique global struct per target region to store kernel properties.
  // This global data structure is used by the offload library to setup the
  // launch parameters.
  void SetTargetKernelProperties(CodeGenModule &CGM, StringRef TargetName,
                                 CGOpenMPRuntimeNVPTX::ExecutionMode Mode,
                                 unsigned ReductionVariableCount,
                                 unsigned ReductionSizeInBytes);

  /// \brief Helper to emit outline 'target' directive. This creates a wrapper
  /// with device specific arguments.
  /// \brief Returns a pointer to the outlined function.
  /// \param D Directive to emit.
  /// \param Name Name of the outlined function.
  /// \param CodeGen Lambda codegen specific to an accelerator device.
  virtual llvm::Function *
  outlineTargetDirective(const OMPExecutableDirective &D, StringRef Name,
                         const RegionCodeGenTy &CodeGen) override;

  /// \brief Emit outlined function specialized for the Fork-Join
  /// programming model for applicable target directives on the NVPTX device.
  /// \param D Directive to emit.
  /// \param TP Kernel properties for this target region.
  /// \param ParentName Name of the function that encloses the target region.
  /// \param OutlinedFn Outlined function value to be defined by this call.
  /// \param OutlinedFnID Outlined function ID value to be defined by this call.
  /// \param IsOffloadEntry True if the outlined function is an offload entry.
  /// An outlined function may not be an entry if, e.g. the if clause always
  /// evaluates to false.
  void emitGenericKernel(const OMPExecutableDirective &D,
                         const TargetKernelProperties &TP, StringRef ParentName,
                         llvm::Function *&OutlinedFn,
                         llvm::Constant *&OutlinedFnID, bool IsOffloadEntry,
                         const RegionCodeGenTy &CodeGen);

  /// \brief Emit outlined function specialized for the Single Program
  /// Multiple Data programming model for applicable target directives on the
  /// NVPTX device.
  /// \param D Directive to emit.
  /// \param TP Kernel properties for this target region.
  /// \param ParentName Name of the function that encloses the target region.
  /// \param OutlinedFn Outlined function value to be defined by this call.
  /// \param OutlinedFnID Outlined function ID value to be defined by this call.
  /// \param IsOffloadEntry True if the outlined function is an offload entry.
  /// An outlined function may not be an entry if, e.g. the if clause always
  /// evaluates to false.
  void emitSPMDKernel(const OMPExecutableDirective &D,
                      const TargetKernelProperties &TP, StringRef ParentName,
                      llvm::Function *&OutlinedFn,
                      llvm::Constant *&OutlinedFnID, bool IsOffloadEntry,
                      const RegionCodeGenTy &CodeGen);

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
                                  bool IsOffloadEntry,
                                  const RegionCodeGenTy &CodeGen,
                                  unsigned CaptureLevel) override;

  /// \brief Emits call to void __kmpc_push_num_threads(ident_t *loc, kmp_int32
  /// global_tid, kmp_int32 num_threads) to generate code for 'num_threads'
  /// clause.
  /// \param NumThreads An integer value of threads.
  virtual void emitNumThreadsClause(CodeGenFunction &CGF,
                                    llvm::Value *NumThreads,
                                    SourceLocation Loc) override;

  /// \brief Emit call to void __kmpc_push_proc_bind(ident_t *loc, kmp_int32
  /// global_tid, int proc_bind) to generate code for 'proc_bind' clause.
  virtual void emitProcBindClause(CodeGenFunction &CGF,
                                  OpenMPProcBindClauseKind ProcBind,
                                  SourceLocation Loc) override;

  /// Call the appropriate runtime routine to notify that we finished
  /// iteration of the dynamic loop.
  ///
  /// \param CGF Reference to current CodeGenFunction.
  /// \param OpenMP Directive.
  /// \param Loc Clang source location.
  /// \param IVSize Size of the iteration variable in bits.
  /// \param IVSigned Sign of the interation variable.
  ///
  virtual void emitForDispatchFinish(CodeGenFunction &CGF,
                                     const OMPLoopDirective &S,
                                     SourceLocation Loc, unsigned IVSize,
                                     bool IVSigned) override;

  /// \brief Emit the code that each thread requires to execute when it
  /// encounters one of the three possible parallelism level. This also emits
  /// the required data sharing code for each level.
  /// \param Level0 Code to emit by the master thread when it encounters a
  /// parallel region.
  /// \param Level1 Code to emit by a worker thread when it encounters a
  /// parallel region.
  /// \param Sequential Code to emit by a worker thread when the parallel region
  /// is to be computed sequentially.
  void emitParallelismLevelCode(
      CodeGenFunction &CGF,
      const llvm::function_ref<llvm::Value *()> &ParallelLevelGen,
      const RegionCodeGenTy &Level0, const RegionCodeGenTy &Level1,
      const RegionCodeGenTy &Sequential);

  /// \brief Emits code for parallel or serial call of the \a OutlinedFn with
  /// variables captured in a record which address is stored in \a
  /// CapturedStruct.
  /// This call is for the Generic Execution Mode.
  /// \param OutlinedFn Outlined function to be run in parallel threads. Type of
  /// this function is void(*)(kmp_int32 *, kmp_int32, struct context_vars*).
  /// \param CapturedVars A pointer to the record with the references to
  /// variables used in \a OutlinedFn function.
  /// \param IfCond Condition in the associated 'if' clause, if it was
  /// specified, nullptr otherwise.
  ///
  void emitGenericParallelCall(CodeGenFunction &CGF, SourceLocation Loc,
                               llvm::Value *OutlinedFn,
                               ArrayRef<llvm::Value *> CapturedVars,
                               const Expr *IfCond);

  /// \brief Emits code for parallel or serial call of the \a OutlinedFn with
  /// variables captured in a record which address is stored in \a
  /// CapturedStruct.
  /// This call is for the SPMD Execution Mode.
  /// \param OutlinedFn Outlined function to be run in parallel threads. Type of
  /// this function is void(*)(kmp_int32 *, kmp_int32, struct context_vars*).
  /// \param CapturedVars A pointer to the record with the references to
  /// variables used in \a OutlinedFn function.
  /// \param IfCond Condition in the associated 'if' clause, if it was
  /// specified, nullptr otherwise.
  ///
  void emitSPMDParallelCall(CodeGenFunction &CGF, SourceLocation Loc,
                            llvm::Value *OutlinedFn,
                            ArrayRef<llvm::Value *> CapturedVars,
                            const Expr *IfCond);

  // \brief Test if a construct is always encountered at nesting level 0.
  bool InL0();

  // \brief Test if a construct is always encountered at nesting level 1.
  bool InL1();

  // \brief Test if a construct is always encountered at nesting level 1 or
  // higher.
  bool InL1Plus();

  // \brief Test if the nesting level at which a construct is encountered is
  // indeterminate.  This happens for orphaned parallel directives.
  bool IndeterminateLevel();

  // \brief Test if we are codegen'ing a target construct in generic or spmd
  // mode.
  bool isSPMDExecutionMode() const;

  // \brief Test if we are codegen'ing a target construct where the OMP runtime
  // has been elided.
  bool isOMPRuntimeInitialized() const;

  /// Register target region related with the launching of Ctor/Dtors entry. On
  /// top of the default registration, an extra global is registered to make
  /// sure SPMD mode is used in the execution of the Ctor/Dtor.
  /// \param DeviceID The device ID of the target region in the system.
  /// \param FileID The file ID of the target region in the system.
  /// \param RegionName The name of the region.
  /// \param Line Line where the declaration the target egion refers to is
  /// defined.
  /// \param Fn The function that implements the target region.
  /// \param IsDtor True if what being registered is a destructor.
  virtual void registerCtorDtorEntry(unsigned DeviceID, unsigned FileID,
                                     StringRef RegionName, unsigned Line,
                                     llvm::Function *Fn, bool IsDtor) override;

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

  /// \brief Emits code for simd call of the \a OutlinedFn with
  /// variables captured in a record which address is stored in \a
  /// CapturedStruct.
  /// \param OutlinedFn Outlined function to be run in simd lanes. Type of
  /// this function is void(*)(kmp_int32 *,  struct context_vars*).
  /// \param CapturedVars A pointer to the record with the references to
  /// variables used in \a OutlinedFn function.
  ///
  void emitSimdCall(CodeGenFunction &CGF, SourceLocation Loc,
                    llvm::Value *OutlinedFn,
                    ArrayRef<llvm::Value *> CapturedVars) override;

  /// \brief Emits a critical region.
  /// \param CriticalName Name of the critical region.
  /// \param CriticalOpGen Generator for the statement associated with the given
  /// critical region.
  /// \param Hint Value of the 'hint' clause (optional).
  void emitCriticalRegion(CodeGenFunction &CGF, StringRef CriticalName,
                          const RegionCodeGenTy &CriticalOpGen,
                          SourceLocation Loc,
                          const Expr *Hint = nullptr) override;

  /// \brief Check if we should generate code as if \a ScheduleKind is static
  /// with a chunk size of 1.
  /// \param ScheduleKind Schedule Kind specified in the 'schedule' clause.
  /// \param ChunkSizeOne True if schedule chunk is one.
  /// \param Ordered true if loop is ordered, false otherwise.
  ///
  bool generateCoalescedSchedule(OpenMPScheduleClauseKind ScheduleKind,
                                 bool ChunkSizeOne,
                                 bool Ordered) const override;

  /// \brief Check if we should generate code as if \a DistScheduleKind is
  /// static non-chunked and \a ScheduleKind is static with a chunk size of 1.
  /// \param DistScheduleKind Schedule Kind specified in the 'dist_schedule'
  /// clause.
  /// \param ScheduleKind Schedule Kind specified in the 'schedule' clause.
  /// \param Chunked True if distribute chunk is specified in the clause.
  /// \param ChunkSizeOne True if schedule chunk is one.
  /// \param Ordered true if loop is ordered, false otherwise.
  ///
  bool generateCoalescedSchedule(OpenMPDistScheduleClauseKind DistScheduleKind,
                                 OpenMPScheduleClauseKind ScheduleKind,
                                 bool DistChunked, bool ChunkSizeOne,
                                 bool Ordered) const override;

  /// \brief Check if we must always generate a barrier at the end of a
  /// particular construct regardless of the presence of a nowait clause.
  /// This may occur when a particular offload device does not support
  /// concurrent execution of certain directive and clause combinations.
  bool requiresBarrier(const OMPLoopDirective &S) const override;

  /// \brief Emit an implicit/explicit barrier for OpenMP threads.
  /// \param Kind Directive for which this implicit barrier call must be
  /// generated. Must be OMPD_barrier for explicit barrier generation.
  /// \param EmitChecks true if need to emit checks for cancellation barriers.
  /// \param ForceSimpleCall true simple barrier call must be emitted, false if
  /// runtime class decides which one to emit (simple or with cancellation
  /// checks).
  ///
  void emitBarrierCall(CodeGenFunction &CGF, SourceLocation Loc,
                       OpenMPDirectiveKind Kind, bool EmitChecks = true,
                       bool ForceSimpleCall = false) override;

  /// \brief This function ought to emit, in the general case, a call to
  // the openmp runtime kmpc_push_num_teams. In NVPTX backend it is not needed
  // as these numbers are obtained through the PTX grid and block configuration.
  /// \param NumTeams An integer expression of teams.
  /// \param ThreadLimit An integer expression of threads.
  void emitNumTeamsClause(CodeGenFunction &CGF, const Expr *NumTeams,
                          const Expr *ThreadLimit, SourceLocation Loc) override;

  /// \brief Emits a BasicBlock "omp.init.ds" for data sharing.
  void emitInitDSBlock(CodeGenFunction &CGF) override;

  /// \brief Emits inlined function for the specified OpenMP teams
  //  directive.
  /// \a D. This outlined function has type void(*)(kmp_int32 *ThreadID,
  /// kmp_int32 BoundID, struct context_vars*).
  /// \param D OpenMP directive.
  /// \param ThreadIDVar Variable for thread id in the current OpenMP region.
  /// \param InnermostKind Kind of innermost directive (for simple directives it
  /// is a directive itself, for combined - its innermost directive).
  /// \param CodeGen Code generation sequence for the \a D directive.
  /// \param CaptureLevel Codegening level of a combined construct.
  llvm::Value *emitTeamsOutlinedFunction(
      const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
      OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen,
      unsigned CaptureLevel = 1, unsigned ImplicitParamStop = 0) override;

  /// \brief Emits inlined function for the specified OpenMP parallel
  //  directive.
  /// \a D. This outlined function has type void(*)(kmp_int32 *ThreadID,
  /// kmp_int32 BoundID, struct context_vars*).
  /// \param D OpenMP directive.
  /// \param ThreadIDVar Variable for thread id in the current OpenMP region.
  /// \param InnermostKind Kind of innermost directive (for simple directives it
  /// is a directive itself, for combined - its innermost directive).
  /// \param CodeGen Code generation sequence for the \a D directive.
  /// \param CaptureLevel Codegening level of a combined construct.
  llvm::Value *emitParallelOutlinedFunction(
      const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
      OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen,
      unsigned CaptureLevel = 1, unsigned ImplicitParamStop = 0) override;

  /// \brief Emits outlined function for the specified OpenMP simd directive
  /// \a D. This outlined function has type void(*)(kmp_int32 *LaneID,
  /// struct context_vars*).
  /// \param D OpenMP directive.
  /// \param LaneIDVar Variable for lane id in the current OpenMP region.
  /// \param InnermostKind Kind of innermost directive (for simple directives it
  /// is a directive itself, for combined - its innermost directive).
  /// \param CodeGen Code generation sequence for the \a D directive.
  llvm::Value *
  emitSimdOutlinedFunction(const OMPExecutableDirective &D,
                           const VarDecl *LaneIDVar, const VarDecl *NumLanesVar,
                           OpenMPDirectiveKind InnermostKind,
                           const RegionCodeGenTy &CodeGen) override;

  /// \brief Emits code for teams call of the \a OutlinedFn with
  /// variables captured in a record which address is stored in \a
  /// CapturedStruct.
  /// \param OutlinedFn Outlined function to be run by team masters. Type of
  /// this function is void(*)(kmp_int32 *, kmp_int32, struct context_vars*).
  /// \param CapturedVars A pointer to the record with the references to
  /// variables used in \a OutlinedFn function.
  ///
  void emitTeamsCall(CodeGenFunction &CGF, const OMPExecutableDirective &D,
                     SourceLocation Loc, llvm::Value *OutlinedFn,
                     ArrayRef<llvm::Value *> CapturedVars) override;

  /// \brief Creates the offloading descriptor in the event any target region
  /// was emitted in the current module and return the function that registers
  /// it. We take advantage of this hook to do data sharing replacements.
  llvm::Function *emitRegistrationFunction() override;

  /// Return false for the current NVPTX OpenMP implementation as it does NOT
  /// supports RTTI.
  bool requiresRTTIDescriptor() override { return false; }

  virtual void emitReduction(CodeGenFunction &CGF, SourceLocation Loc,
                             ArrayRef<const Expr *> Privates,
                             ArrayRef<const Expr *> LHSExprs,
                             ArrayRef<const Expr *> RHSExprs,
                             ArrayRef<const Expr *> ReductionOps,
                             bool WithNowait, bool SimpleReduction,
                             OpenMPDirectiveKind ReductionKind) override;

  /// Translates the native parameter of outlined function if this is required
  /// for target.
  /// \param FD Field decl from captured record for the paramater.
  /// \param NativeParam Parameter itself.
  const VarDecl *translateParameter(const FieldDecl *FD,
                                    const VarDecl *NativeParam) const override;

  /// Gets the address of the native argument basing on the address of the
  /// target-specific parameter.
  /// \param NativeParam Parameter itself.
  /// \param TargetParam Corresponding target-specific parameter.
  Address getParameterAddress(CodeGenFunction &CGF, const VarDecl *NativeParam,
                              const VarDecl *TargetParam) const override;

  /// Emits call of the outlined function with the provided arguments.
  void emitOutlinedFunctionCall(
      CodeGenFunction &CGF, SourceLocation Loc, llvm::Value *OutlinedFn,
      ArrayRef<llvm::Value *> Args = llvm::None) const override;
};

} // CodeGen namespace.
} // clang namespace.

#endif // LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMENVPTX_H
