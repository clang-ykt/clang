//===--- CGStmtOpenMP.cpp - Emit LLVM Code from Statements ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit OpenMP nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CGCleanup.h"
#include "CGOpenMPRuntime.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "TargetInfo.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/DeclOpenMP.h"
#include "llvm/IR/CallSite.h"
using namespace clang;
using namespace CodeGen;

namespace {
/// Lexical scope for OpenMP executable constructs, that handles correct codegen
/// for captured expressions.
class OMPLexicalScope final : public CodeGenFunction::LexicalScope {
  void emitPreInitStmt(CodeGenFunction &CGF, const OMPExecutableDirective &S) {
    for (const auto *C : S.clauses()) {
      if (auto *CPI = OMPClauseWithPreInit::get(C)) {
        if (auto *PreInit = cast_or_null<DeclStmt>(CPI->getPreInitStmt())) {
          for (const auto *I : PreInit->decls()) {
            if (!I->hasAttr<OMPCaptureNoInitAttr>())
              CGF.EmitVarDecl(cast<VarDecl>(*I));
            else {
              CodeGenFunction::AutoVarEmission Emission =
                  CGF.EmitAutoVarAlloca(cast<VarDecl>(*I));
              CGF.EmitAutoVarCleanups(Emission);
            }
          }
        }
      }
    }
  }
  CodeGenFunction::OMPPrivateScope InlinedShareds;

  static bool isCapturedVar(CodeGenFunction &CGF, const VarDecl *VD) {
    return CGF.LambdaCaptureFields.lookup(VD) ||
           (CGF.CapturedStmtInfo && CGF.CapturedStmtInfo->lookup(VD)) ||
           (CGF.CurCodeDecl && isa<BlockDecl>(CGF.CurCodeDecl));
  }

public:
  OMPLexicalScope(CodeGenFunction &CGF, const OMPExecutableDirective &S,
                  bool AsInlined = false, bool EmitPreInit = true)
      : CodeGenFunction::LexicalScope(CGF, S.getSourceRange()),
        InlinedShareds(CGF) {
    if (EmitPreInit)
      emitPreInitStmt(CGF, S);
    if (AsInlined) {
      if (S.hasAssociatedStmt()) {
        auto *CS = cast<CapturedStmt>(S.getAssociatedStmt());
        for (auto &C : CS->captures()) {
          if (C.capturesVariable() || C.capturesVariableByCopy()) {
            auto *VD = C.getCapturedVar();
            assert(VD == VD->getCanonicalDecl() &&
                        "Canonical decl must be captured.");
            DeclRefExpr DRE(const_cast<VarDecl *>(VD),
                            isCapturedVar(CGF, VD) ||
                                (CGF.CapturedStmtInfo &&
                                 InlinedShareds.isGlobalVarCaptured(VD)),
                            VD->getType().getNonReferenceType(), VK_LValue,
                            C.getLocation());
            InlinedShareds.addPrivate(VD, [&CGF, &DRE]() -> Address {
              return CGF.EmitLValue(&DRE).getAddress();
            });
          }
        }
        (void)InlinedShareds.Privatize();
      }
    }
  }
};

/// Private scope for OpenMP loop-based directives, that supports capturing
/// of used expression from loop statement.
class OMPLoopScope : public CodeGenFunction::RunCleanupsScope {
  void emitPreInitStmt(CodeGenFunction &CGF, const OMPLoopDirective &S) {
    CodeGenFunction::OMPPrivateScope PreCondScope(CGF);
    for (auto *E : S.counters()) {
      const auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
      (void)PreCondScope.addPrivate(VD, [&CGF, VD]() {
        return CGF.CreateMemTemp(VD->getType().getNonReferenceType());
      });
    }
    (void)PreCondScope.Privatize();
    if (auto *LD = dyn_cast<OMPLoopDirective>(&S)) {
      if (auto *PreInits = cast_or_null<DeclStmt>(LD->getPreInits())) {
        for (const auto *I : PreInits->decls())
          CGF.EmitOMPHelperVar(cast<VarDecl>(I));
      }
    }
  }

public:
  OMPLoopScope(CodeGenFunction &CGF, const OMPLoopDirective &S,
               bool EmitPreInit)
      : CodeGenFunction::RunCleanupsScope(CGF) {
    if (EmitPreInit)
      emitPreInitStmt(CGF, S);
  }
};

/// Lexical scope for OpenMP parallel construct, that handles correct codegen
/// for captured expressions.
class OMPParallelScope final : public CodeGenFunction::LexicalScope {
  void emitPreInitStmt(CodeGenFunction &CGF, const OMPExecutableDirective &S) {
    OpenMPDirectiveKind Kind = S.getDirectiveKind();
    if (!(isOpenMPTargetExecutionDirective(Kind) ||
          isOpenMPLoopBoundSharingDirective(Kind)) &&
        isOpenMPParallelDirective(Kind)) {
      for (const auto *C : S.clauses()) {
        if (auto *CPI = OMPClauseWithPreInit::get(C)) {
          if (auto *PreInit = cast_or_null<DeclStmt>(CPI->getPreInitStmt())) {
            for (const auto *I : PreInit->decls()) {
              if (!I->hasAttr<OMPCaptureNoInitAttr>())
                CGF.EmitVarDecl(cast<VarDecl>(*I));
              else {
                CodeGenFunction::AutoVarEmission Emission =
                    CGF.EmitAutoVarAlloca(cast<VarDecl>(*I));
                CGF.EmitAutoVarCleanups(Emission);
              }
            }
          }
        }
      }
    }
  }

public:
  OMPParallelScope(CodeGenFunction &CGF, const OMPExecutableDirective &S)
      : CodeGenFunction::LexicalScope(CGF, S.getSourceRange()) {
    emitPreInitStmt(CGF, S);
  }
};

/// Lexical scope for OpenMP teams construct, that handles correct codegen
/// for captured expressions.
class OMPTeamsScope final : public CodeGenFunction::LexicalScope {
  void emitPreInitStmt(CodeGenFunction &CGF, const OMPExecutableDirective &S) {
    OpenMPDirectiveKind Kind = S.getDirectiveKind();
    if (!isOpenMPTargetExecutionDirective(Kind) &&
        isOpenMPTeamsDirective(Kind)) {
      for (const auto *C : S.clauses()) {
        if (auto *CPI = OMPClauseWithPreInit::get(C)) {
          if (auto *PreInit = cast_or_null<DeclStmt>(CPI->getPreInitStmt())) {
            for (const auto *I : PreInit->decls()) {
              if (!I->hasAttr<OMPCaptureNoInitAttr>())
                CGF.EmitVarDecl(cast<VarDecl>(*I));
              else {
                CodeGenFunction::AutoVarEmission Emission =
                    CGF.EmitAutoVarAlloca(cast<VarDecl>(*I));
                CGF.EmitAutoVarCleanups(Emission);
              }
            }
          }
        }
      }
    }
  }

public:
  OMPTeamsScope(CodeGenFunction &CGF, const OMPExecutableDirective &S)
      : CodeGenFunction::LexicalScope(CGF, S.getSourceRange()) {
    emitPreInitStmt(CGF, S);
  }
};

} // namespace

LValue CodeGenFunction::EmitOMPSharedLValue(const Expr *E) {
  if (auto *OrigDRE = dyn_cast<DeclRefExpr>(E)) {
    if (auto *OrigVD = dyn_cast<VarDecl>(OrigDRE->getDecl())) {
      OrigVD = OrigVD->getCanonicalDecl();
      bool IsCaptured =
          LambdaCaptureFields.lookup(OrigVD) ||
          (CapturedStmtInfo && CapturedStmtInfo->lookup(OrigVD)) ||
          (CurCodeDecl && isa<BlockDecl>(CurCodeDecl));
      DeclRefExpr DRE(const_cast<VarDecl *>(OrigVD), IsCaptured,
                      OrigDRE->getType(), VK_LValue, OrigDRE->getExprLoc());
      return EmitLValue(&DRE);
    }
  }
  return EmitLValue(E);
}

llvm::Value *CodeGenFunction::getTypeSize(QualType Ty) {
  auto &C = getContext();
  llvm::Value *Size = nullptr;
  auto SizeInChars = C.getTypeSizeInChars(Ty);
  if (SizeInChars.isZero()) {
    // getTypeSizeInChars() returns 0 for a VLA.
    while (auto *VAT = C.getAsVariableArrayType(Ty)) {
      llvm::Value *ArraySize;
      std::tie(ArraySize, Ty) = getVLASize(VAT);
      Size = Size ? Builder.CreateNUWMul(Size, ArraySize) : ArraySize;
    }
    SizeInChars = C.getTypeSizeInChars(Ty);
    if (SizeInChars.isZero())
      return llvm::ConstantInt::get(SizeTy, /*V=*/0);
    Size = Builder.CreateNUWMul(Size, CGM.getSize(SizeInChars));
  } else
    Size = CGM.getSize(SizeInChars);
  return Size;
}

static QualType getCanonicalParamType(ASTContext &C, QualType T) {
  if (T->isLValueReferenceType()) {
    return C.getLValueReferenceType(
        getCanonicalParamType(C, T.getNonReferenceType()),
        /*SpelledAsLValue=*/false);
  }
  if (T->isPointerType())
    return C.getPointerType(getCanonicalParamType(C, T->getPointeeType()));
  if (auto *A = T->getAsArrayTypeUnsafe()) {
    if (auto *VLA = dyn_cast<VariableArrayType>(A))
      return getCanonicalParamType(C, VLA->getElementType());
    else if (!A->isVariablyModifiedType())
      return C.getCanonicalType(T);
  }
  return C.getCanonicalParamType(T);
}

void CodeGenFunction::GenerateOpenMPCapturedVars(
    const CapturedStmt &S, SmallVectorImpl<llvm::Value *> &CapturedVars,
    unsigned CaptureLevel) {
  const RecordDecl *RD = S.getCapturedRecordDecl();
  auto CurField = RD->field_begin();
  auto CurCap = S.captures().begin();
  for (CapturedStmt::const_capture_init_iterator I = S.capture_init_begin(),
                                                 E = S.capture_init_end();
       I != E; ++I, ++CurField, ++CurCap) {
    if (CurCap->capturesVariable() || CurCap->capturesVariableByCopy()) {
      auto *Var = CurCap->getCapturedVar();
      if (auto *C = dyn_cast<OMPCapturedExprDecl>(Var))
        // Check to see if the capture is to be a parameter in the
        // outlined function at this level.
        if (C->getCaptureLevel() < CaptureLevel)
          continue;
    }

    if (CurField->hasCapturedVLAType()) {
      auto VAT = CurField->getCapturedVLAType();
      auto *Val = VLASizeMap[VAT->getSizeExpr()];
      CapturedVars.push_back(Val);
    } else if (CurCap->capturesThis())
      CapturedVars.push_back(CXXThisValue);
    else if (CurCap->capturesVariableByCopy()) {
      llvm::Value *CV = EmitLoadOfScalar(EmitLValue(*I), CurCap->getLocation());

      // If the field is not a pointer, we need to save the actual value
      // and load it as a void pointer.
      if (!CurField->getType()->isAnyPointerType()) {
        auto &Ctx = getContext();
        auto DstAddr = CreateMemTemp(
            Ctx.getUIntPtrType(),
            Twine(CurCap->getCapturedVar()->getName()) + ".casted");
        LValue DstLV = MakeAddrLValue(DstAddr, Ctx.getUIntPtrType());

        auto *SrcAddrVal = EmitScalarConversion(
            DstAddr.getPointer(), Ctx.getPointerType(Ctx.getUIntPtrType()),
            Ctx.getPointerType(CurField->getType()), CurCap->getLocation());
        LValue SrcLV =
            MakeNaturalAlignAddrLValue(SrcAddrVal, CurField->getType());

        // Store the value using the source type pointer.
        EmitStoreThroughLValue(RValue::get(CV), SrcLV);

        // Load the value using the destination type pointer.
        CV = EmitLoadOfScalar(DstLV, CurCap->getLocation());
      }
      CapturedVars.push_back(CV);
    } else {
      assert(CurCap->capturesVariable() && "Expected capture by reference.");
      CapturedVars.push_back(EmitLValue(*I).getAddress().getPointer());
    }
  }
}

static Address castValueFromUintptr(CodeGenFunction &CGF, QualType DstType,
                                    StringRef Name, LValue AddrLV,
                                    SourceLocation Loc,
                                    bool isReferenceType = false) {
  ASTContext &Ctx = CGF.getContext();

  auto *CastedPtr = CGF.EmitScalarConversion(AddrLV.getAddress().getPointer(),
                                             Ctx.getUIntPtrType(),
                                             Ctx.getPointerType(DstType), Loc);
  auto TmpAddr =
      CGF.MakeNaturalAlignAddrLValue(CastedPtr, Ctx.getPointerType(DstType))
          .getAddress();

  // If we are dealing with references we need to return the address of the
  // reference instead of the reference of the value.
  if (isReferenceType) {
    QualType RefType = Ctx.getLValueReferenceType(DstType);
    auto *RefVal = TmpAddr.getPointer();
    TmpAddr = CGF.CreateMemTemp(RefType, Twine(Name) + ".ref");
    auto TmpLVal = CGF.MakeAddrLValue(TmpAddr, RefType);
    CGF.EmitStoreOfScalar(RefVal, TmpLVal);
  }

  return TmpAddr;
}

namespace {
  /// Contains required data for proper outlined function codegen.
  struct FunctionOptions {
    /// Captured statement for which the function is generated.
    const CapturedStmt *S = nullptr;
    /// Should function use only captured fields for codegen.
    const bool UseCapturedArgumentsOnly = false;
    /// Level of captured function.
    const unsigned CaptureLevel = 0;
    /// Index of the implicit parameter.
    const unsigned ImplicitParamStop = 0;
    /// Arguments are not aliased.
    const bool NonAliasedMaps = true;
    /// true if cast to/from  UIntPtr is required for variables captured by
    /// value.
    const bool UIntPtrCastRequired = true;
    /// true if only casted arguments must be registered as local args or VLA
    /// sizes.
    const bool RegisterCastedArgsOnly = false;
    /// Name of the generated function.
    const StringRef FunctionName;
    explicit FunctionOptions(const CapturedStmt *S,
                             bool UseCapturedArgumentsOnly,
                             unsigned CaptureLevel, unsigned ImplicitParamStop,
                             bool NonAliasedMaps, bool UIntPtrCastRequired,
                             bool RegisterCastedArgsOnly,
                             StringRef FunctionName)
        : S(S), UseCapturedArgumentsOnly(UseCapturedArgumentsOnly),
          CaptureLevel(CaptureLevel), ImplicitParamStop(ImplicitParamStop),
          NonAliasedMaps(NonAliasedMaps),
          UIntPtrCastRequired(UIntPtrCastRequired),
          RegisterCastedArgsOnly(UIntPtrCastRequired && RegisterCastedArgsOnly),
          FunctionName(FunctionName) {}
  };
}

static llvm::Function *emitOutlinedFunctionPrologue(
    CodeGenFunction &CGF, FunctionArgList &Args,
    llvm::MapVector<const Decl *, std::pair<const VarDecl *, Address>>
        &LocalAddrs,
    llvm::DenseMap<const Decl *, std::pair<const Expr *, llvm::Value *>>
        &VLASizes,
    llvm::Value *&CXXThisValue, const FunctionOptions &FO) {
  const CapturedDecl *CD = FO.S->getCapturedDecl();
  const RecordDecl *RD = FO.S->getCapturedRecordDecl();
  assert(CD->hasBody() && "missing CapturedDecl body");

  CXXThisValue = nullptr;
  // Build the argument list.
  CodeGenModule &CGM = CGF.CGM;
  ASTContext &Ctx = CGM.getContext();

  CGF.GenerateOpenMPCapturedStmtParameters(
      *(FO.S), FO.UseCapturedArgumentsOnly, FO.CaptureLevel,
      FO.ImplicitParamStop, FO.NonAliasedMaps, FO.UIntPtrCastRequired, Args);

  unsigned ImplicitParamStop = FO.ImplicitParamStop == 0
                                   ? CD->getContextParamPosition()
                                   : FO.ImplicitParamStop;
  FunctionArgList TargetArgs;
  if (FO.UIntPtrCastRequired) {
    TargetArgs.append(Args.begin(), Args.end());
  } else {
    if (!FO.UseCapturedArgumentsOnly) {
      TargetArgs.append(CD->param_begin(),
                        std::next(CD->param_begin(), ImplicitParamStop));
    }
    auto I = FO.S->captures().begin();
    unsigned Cnt = FO.UseCapturedArgumentsOnly ? 0 : ImplicitParamStop;
    for (auto *FD : RD->fields()) {
      if (I->capturesVariable() || I->capturesVariableByCopy()) {
        VarDecl *CapVar = I->getCapturedVar();
        if (auto *C = dyn_cast<OMPCapturedExprDecl>(CapVar)) {
          // Check to see if the capture is to be a parameter in the
          // outlined function at this level.
          if (C->getCaptureLevel() < FO.CaptureLevel) {
            ++I;
            continue;
          }
        }
      }
      TargetArgs.emplace_back(
          CGM.getOpenMPRuntime().translateParameter(FD, Args[Cnt]));
      ++I;
      ++Cnt;
    }
    TargetArgs.append(
        std::next(CD->param_begin(), CD->getContextParamPosition() + 1),
        CD->param_end());
  }

  // Create the function declaration.
  FunctionType::ExtInfo ExtInfo;
  const CGFunctionInfo &FuncInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(Ctx.VoidTy, TargetArgs);
  llvm::FunctionType *FuncLLVMTy = CGM.getTypes().GetFunctionType(FuncInfo);

  llvm::Function *F =
      llvm::Function::Create(FuncLLVMTy, llvm::GlobalValue::InternalLinkage,
                             FO.FunctionName, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(CD, F, FuncInfo);
  if (CD->isNothrow())
    F->setDoesNotThrow();

  // Generate the function.
  CGF.StartFunction(CD, Ctx.VoidTy, F, FuncInfo, TargetArgs,
                    FO.S->getLocStart(), CD->getBody()->getLocStart());
  unsigned Cnt = FO.UseCapturedArgumentsOnly ? 0 : ImplicitParamStop;
  auto I = FO.S->captures().begin();
  for (auto *FD : RD->fields()) {
    if (I->capturesVariable() || I->capturesVariableByCopy()) {
      auto *Var = I->getCapturedVar();
      if (auto *C = dyn_cast<OMPCapturedExprDecl>(Var)) {
        // Check to see if the capture is to be a parameter in the
        // outlined function at this level.
        if (C->getCaptureLevel() < FO.CaptureLevel) {
          ++I;
          continue;
        }
      }
    }

    // Do not map arguments if we emit function with non-original types.
    Address LocalAddr(Address::invalid());
    if (!FO.UIntPtrCastRequired && Args[Cnt] != TargetArgs[Cnt]) {
      LocalAddr = CGM.getOpenMPRuntime().getParameterAddress(CGF, Args[Cnt],
                                                             TargetArgs[Cnt]);
    } else {
      LocalAddr = CGF.GetAddrOfLocalVar(Args[Cnt]);
    }
    // If we are capturing a pointer by copy we don't need to do anything, just
    // use the value that we get from the arguments.
    if (I->capturesVariableByCopy() && FD->getType()->isAnyPointerType()) {
      const VarDecl *CurVD = I->getCapturedVar();
      // If the variable is a reference we need to materialize it here.
      if (CurVD->getType()->isReferenceType()) {
        Address RefAddr = CGF.CreateMemTemp(
            CurVD->getType(), CGF.getPointerAlign(), ".materialized_ref");
        CGF.EmitStoreOfScalar(LocalAddr.getPointer(), RefAddr,
                              /*Volatile=*/false, CurVD->getType());
        LocalAddr = RefAddr;
      }
      if (!FO.RegisterCastedArgsOnly)
        LocalAddrs.insert({Args[Cnt], {CurVD, LocalAddr}});
      ++Cnt;
      ++I;
      continue;
    }

    LValue ArgLVal = CGF.MakeAddrLValue(LocalAddr, Args[Cnt]->getType(),
                                        AlignmentSource::Decl);
    if (FD->hasCapturedVLAType()) {
      if (FO.UIntPtrCastRequired) {
        ArgLVal = CGF.MakeAddrLValue(
            castValueFromUintptr(CGF, FD->getType(), Args[Cnt]->getName(),
                                 ArgLVal, I->getLocation()),
            FD->getType(), AlignmentSource::Decl);
      }
      auto *ExprArg = CGF.EmitLoadOfScalar(ArgLVal, I->getLocation());
      auto VAT = FD->getCapturedVLAType();
      VLASizes.insert({Args[Cnt], {VAT->getSizeExpr(), ExprArg}});
    } else if (I->capturesVariable()) {
      auto *Var = I->getCapturedVar();
      QualType VarTy = Var->getType();
      Address ArgAddr = ArgLVal.getAddress();
      if (!VarTy->isReferenceType()) {
        if (ArgLVal.getType()->isLValueReferenceType()) {
          ArgAddr = CGF.EmitLoadOfReference(
              ArgAddr, ArgLVal.getType()->castAs<ReferenceType>());
        } else if (!VarTy->isVariablyModifiedType() || !VarTy->isPointerType()) {
          assert(ArgLVal.getType()->isPointerType());
          ArgAddr = CGF.EmitLoadOfPointer(
              ArgAddr, ArgLVal.getType()->castAs<PointerType>());
        }
      }
      if (!FO.RegisterCastedArgsOnly) {
        LocalAddrs.insert(
            {Args[Cnt],
             {Var, Address(ArgAddr.getPointer(), Ctx.getDeclAlign(Var))}});
      }
    } else if (I->capturesVariableByCopy()) {
      assert(!FD->getType()->isAnyPointerType() &&
             "Not expecting a captured pointer.");
      auto *Var = I->getCapturedVar();
      QualType VarTy = Var->getType();
      LocalAddrs.insert(
          {Args[Cnt],
           {Var, FO.UIntPtrCastRequired
                     ? castValueFromUintptr(
                           CGF, FD->getType(), Args[Cnt]->getName(), ArgLVal,
                           I->getLocation(), VarTy->isReferenceType())
                     : ArgLVal.getAddress()}});
    } else {
      // If 'this' is captured, load it into CXXThisValue.
      assert(I->capturesThis());
      CXXThisValue = CGF.EmitLoadOfScalar(ArgLVal, Args[Cnt]->getLocation());
      LocalAddrs.insert({Args[Cnt], {nullptr, ArgLVal.getAddress()}});
    }
    ++Cnt;
    ++I;
  }

  return F;
}

llvm::Function *CodeGenFunction::GenerateOpenMPCapturedStmtFunction(
    const CapturedStmt &S, bool UseCapturedArgumentsOnly, unsigned CaptureLevel,
    unsigned ImplicitParamStop, bool NonAliasedMaps) {
  assert(
      CapturedStmtInfo &&
      "CapturedStmtInfo should be set when generating the captured function");
  const CapturedDecl *CD = S.getCapturedDecl();
  // Build the argument list.
  bool NeedWrapperFunction =
      getDebugInfo() &&
      CGM.getCodeGenOpts().getDebugInfo() >= codegenoptions::LimitedDebugInfo;
  FunctionArgList Args;
  llvm::MapVector<const Decl *, std::pair<const VarDecl *, Address>> LocalAddrs;
  llvm::DenseMap<const Decl *, std::pair<const Expr *, llvm::Value *>> VLASizes;
  SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  Out << CapturedStmtInfo->getHelperName();
  if (NeedWrapperFunction)
    Out << "_debug__";
  FunctionOptions FO(&S, UseCapturedArgumentsOnly, CaptureLevel,
                     ImplicitParamStop, NonAliasedMaps, !NeedWrapperFunction,
                     /*RegisterCastedArgsOnly=*/false, Out.str());
  llvm::Function *F = emitOutlinedFunctionPrologue(*this, Args, LocalAddrs,
                                                   VLASizes, CXXThisValue, FO);
  for (const auto &LocalAddrPair : LocalAddrs) {
    if (LocalAddrPair.second.first) {
      setAddrOfLocalVar(LocalAddrPair.second.first,
                        LocalAddrPair.second.second);
    }
  }
  for (const auto &VLASizePair : VLASizes)
    VLASizeMap[VLASizePair.second.first] = VLASizePair.second.second;
  PGO.assignRegionCounters(GlobalDecl(CD), F);
  CapturedStmtInfo->EmitBody(*this, CD->getBody());
  FinishFunction(CD->getBodyRBrace());
  if (!NeedWrapperFunction)
    return F;

  FunctionOptions WrapperFO(&S, UseCapturedArgumentsOnly, CaptureLevel,
                            ImplicitParamStop, NonAliasedMaps,
                            /*UIntPtrCastRequired=*/true,
                            /*RegisterCastedArgsOnly=*/true,
                            CapturedStmtInfo->getHelperName());
  CodeGenFunction WrapperCGF(CGM, /*suppressNewContext=*/true);
  Args.clear();
  LocalAddrs.clear();
  VLASizes.clear();
  llvm::Function *WrapperF =
      emitOutlinedFunctionPrologue(WrapperCGF, Args, LocalAddrs, VLASizes,
                                   WrapperCGF.CXXThisValue, WrapperFO);
  llvm::SmallVector<llvm::Value *, 4> CallArgs;
  for (const auto *Arg : Args) {
    llvm::Value *CallArg;
    auto I = LocalAddrs.find(Arg);
    if (I != LocalAddrs.end()) {
      LValue LV = WrapperCGF.MakeAddrLValue(
          I->second.second,
          I->second.first ? I->second.first->getType() : Arg->getType(),
          AlignmentSource::Decl);
      CallArg = WrapperCGF.EmitLoadOfScalar(LV, S.getLocStart());
    } else {
      auto EI = VLASizes.find(Arg);
      if (EI != VLASizes.end())
        CallArg = EI->second.second;
      else {
        LValue LV =
            WrapperCGF.MakeAddrLValue(WrapperCGF.GetAddrOfLocalVar(Arg),
                                      Arg->getType(), AlignmentSource::Decl);
        CallArg = WrapperCGF.EmitLoadOfScalar(LV, S.getLocStart());
      }
    }
    CallArgs.emplace_back(WrapperCGF.EmitFromMemory(CallArg, Arg->getType()));
  }
  CGM.getOpenMPRuntime().emitOutlinedFunctionCall(WrapperCGF, S.getLocStart(),
                                                  F, CallArgs);
  WrapperCGF.FinishFunction();
  return WrapperF;
}

void CodeGenFunction::GenerateOpenMPCapturedStmtParameters(
    const CapturedStmt &S, bool UseCapturedArgumentsOnly, unsigned CaptureLevel,
    unsigned ImplicitParamStop, bool NonAliasedMaps, bool UIntPtrCastRequired,
    FunctionArgList &Args) {
  const CapturedDecl *CD = S.getCapturedDecl();
  const RecordDecl *RD = S.getCapturedRecordDecl();
  assert(CD->hasBody() && "missing CapturedDecl body");

  // Build the argument list.
  ASTContext &Ctx = CGM.getContext();
  if (ImplicitParamStop == 0)
    ImplicitParamStop = CD->getContextParamPosition();
  if (!UseCapturedArgumentsOnly) {
    Args.append(CD->param_begin(),
                std::next(CD->param_begin(), ImplicitParamStop));
  }
  FunctionDecl *DebugFunctionDecl = nullptr;
  if (!UIntPtrCastRequired) {
    FunctionProtoType::ExtProtoInfo EPI;
    DebugFunctionDecl = FunctionDecl::Create(
        Ctx, Ctx.getTranslationUnitDecl(), S.getLocStart(), SourceLocation(),
        DeclarationName(), Ctx.VoidTy,
        Ctx.getTrivialTypeSourceInfo(
            Ctx.getFunctionType(Ctx.VoidTy, llvm::None, EPI)),
        SC_Static, /*isInlineSpecified=*/false, /*hasWrittenPrototype=*/false);
  }
  auto I = S.captures().begin();
  for (auto *FD : RD->fields()) {
    QualType ArgType = FD->getType();
    IdentifierInfo *II = nullptr;
    VarDecl *CapVar = nullptr;

    if (I->capturesVariable() || I->capturesVariableByCopy()) {
      CapVar = I->getCapturedVar();
      if (auto *C = dyn_cast<OMPCapturedExprDecl>(CapVar)) {
        // Check to see if the capture is to be a parameter in the
        // outlined function at this level.
        if (C->getCaptureLevel() < CaptureLevel) {
          ++I;
          continue;
        }
      }
    }

    // If this is a capture by copy and the type is not a pointer, the outlined
    // function argument type should be uintptr and the value properly casted to
    // uintptr. This is necessary given that the runtime library is only able to
    // deal with pointers. We can pass in the same way the VLA type sizes to the
    // outlined function.
    if ((I->capturesVariableByCopy() && !ArgType->isAnyPointerType()) ||
        I->capturesVariableArrayType()) {
      if (UIntPtrCastRequired)
        ArgType = Ctx.getUIntPtrType();
    }

    if (I->capturesVariable() || I->capturesVariableByCopy()) {
      CapVar = I->getCapturedVar();
      II = CapVar->getIdentifier();
    } else if (I->capturesThis())
      II = &Ctx.Idents.get("this");
    else {
      assert(I->capturesVariableArrayType());
      II = &Ctx.Idents.get("vla");
    }
    if (ArgType->isVariablyModifiedType())
      ArgType = getCanonicalParamType(Ctx, ArgType);
    if (NonAliasedMaps &&
        (ArgType->isAnyPointerType() || ArgType->isReferenceType()))
      ArgType = ArgType.withRestrict();
    VarDecl *Arg;
    if (DebugFunctionDecl && (CapVar || I->capturesThis())) {
      Arg = ParmVarDecl::Create(
          Ctx, DebugFunctionDecl,
          CapVar ? CapVar->getLocStart() : FD->getLocStart(),
          CapVar ? CapVar->getLocation() : FD->getLocation(), II, ArgType,
          /*TInfo=*/nullptr, SC_None, /*DefArg=*/nullptr);
    } else {
      Arg = ImplicitParamDecl::Create(Ctx, /*DC=*/nullptr, FD->getLocation(),
                                      II, ArgType, ImplicitParamDecl::Other);
    }
    Args.emplace_back(Arg);
    ++I;
  }
  Args.append(std::next(CD->param_begin(), CD->getContextParamPosition() + 1),
              CD->param_end());
}

//===----------------------------------------------------------------------===//
//                              OpenMP Directive Emission
//===----------------------------------------------------------------------===//
void CodeGenFunction::EmitOMPAggregateAssign(
    Address DestAddr, Address SrcAddr, QualType OriginalType,
    const llvm::function_ref<void(Address, Address)> &CopyGen) {
  // Perform element-by-element initialization.
  QualType ElementTy;

  // Drill down to the base element type on both arrays.
  auto ArrayTy = OriginalType->getAsArrayTypeUnsafe();
  auto NumElements = emitArrayLength(ArrayTy, ElementTy, DestAddr);
  SrcAddr = Builder.CreateElementBitCast(SrcAddr, DestAddr.getElementType());

  auto SrcBegin = SrcAddr.getPointer();
  auto DestBegin = DestAddr.getPointer();
  // Cast from pointer to array type to pointer to single element.
  auto DestEnd = Builder.CreateGEP(DestBegin, NumElements);
  // The basic structure here is a while-do loop.
  auto BodyBB = createBasicBlock("omp.arraycpy.body");
  auto DoneBB = createBasicBlock("omp.arraycpy.done");
  auto IsEmpty =
      Builder.CreateICmpEQ(DestBegin, DestEnd, "omp.arraycpy.isempty");
  Builder.CreateCondBr(IsEmpty, DoneBB, BodyBB);

  // Enter the loop body, making that address the current address.
  auto EntryBB = Builder.GetInsertBlock();
  EmitBlock(BodyBB);

  CharUnits ElementSize = getContext().getTypeSizeInChars(ElementTy);

  llvm::PHINode *SrcElementPHI =
    Builder.CreatePHI(SrcBegin->getType(), 2, "omp.arraycpy.srcElementPast");
  SrcElementPHI->addIncoming(SrcBegin, EntryBB);
  Address SrcElementCurrent =
      Address(SrcElementPHI,
              SrcAddr.getAlignment().alignmentOfArrayElement(ElementSize));

  llvm::PHINode *DestElementPHI =
    Builder.CreatePHI(DestBegin->getType(), 2, "omp.arraycpy.destElementPast");
  DestElementPHI->addIncoming(DestBegin, EntryBB);
  Address DestElementCurrent =
    Address(DestElementPHI,
            DestAddr.getAlignment().alignmentOfArrayElement(ElementSize));

  // Emit copy.
  CopyGen(DestElementCurrent, SrcElementCurrent);

  // Shift the address forward by one element.
  auto DestElementNext = Builder.CreateConstGEP1_32(
      DestElementPHI, /*Idx0=*/1, "omp.arraycpy.dest.element");
  auto SrcElementNext = Builder.CreateConstGEP1_32(
      SrcElementPHI, /*Idx0=*/1, "omp.arraycpy.src.element");
  // Check whether we've reached the end.
  auto Done =
      Builder.CreateICmpEQ(DestElementNext, DestEnd, "omp.arraycpy.done");
  Builder.CreateCondBr(Done, DoneBB, BodyBB);
  DestElementPHI->addIncoming(DestElementNext, Builder.GetInsertBlock());
  SrcElementPHI->addIncoming(SrcElementNext, Builder.GetInsertBlock());

  // Done.
  EmitBlock(DoneBB, /*IsFinished=*/true);
}

void CodeGenFunction::EmitOMPCopy(QualType OriginalType, Address DestAddr,
                                  Address SrcAddr, const VarDecl *DestVD,
                                  const VarDecl *SrcVD, const Expr *Copy) {
  if (OriginalType->isArrayType()) {
    auto *BO = dyn_cast<BinaryOperator>(Copy);
    if (BO && BO->getOpcode() == BO_Assign) {
      // Perform simple memcpy for simple copying.
      EmitAggregateAssign(DestAddr, SrcAddr, OriginalType);
    } else {
      // For arrays with complex element types perform element by element
      // copying.
      EmitOMPAggregateAssign(
          DestAddr, SrcAddr, OriginalType,
          [this, Copy, SrcVD, DestVD](Address DestElement, Address SrcElement) {
            // Working with the single array element, so have to remap
            // destination and source variables to corresponding array
            // elements.
            CodeGenFunction::OMPPrivateScope Remap(*this);
            Remap.addPrivate(DestVD, [DestElement]() -> Address {
              return DestElement;
            });
            Remap.addPrivate(
                SrcVD, [SrcElement]() -> Address { return SrcElement; });
            (void)Remap.Privatize();
            EmitIgnoredExpr(Copy);
          });
    }
  } else {
    // Remap pseudo source variable to private copy.
    CodeGenFunction::OMPPrivateScope Remap(*this);
    Remap.addPrivate(SrcVD, [SrcAddr]() -> Address { return SrcAddr; });
    Remap.addPrivate(DestVD, [DestAddr]() -> Address { return DestAddr; });
    (void)Remap.Privatize();
    // Emit copying of the whole variable.
    EmitIgnoredExpr(Copy);
  }
}

bool CodeGenFunction::EmitOMPFirstprivateClause(const OMPExecutableDirective &D,
                                                OMPPrivateScope &PrivateScope) {
  if (!HaveInsertPoint())
    return false;
  bool FirstprivateIsLastprivate = false;
  llvm::DenseSet<const VarDecl *> Lastprivates;
  for (const auto *C : D.getClausesOfKind<OMPLastprivateClause>()) {
    for (const auto *D : C->varlists())
      Lastprivates.insert(
          cast<VarDecl>(cast<DeclRefExpr>(D)->getDecl())->getCanonicalDecl());
  }
  llvm::DenseSet<const VarDecl *> EmittedAsFirstprivate;
  CGCapturedStmtInfo CapturesInfo(cast<CapturedStmt>(*D.getAssociatedStmt()));
  for (const auto *C : D.getClausesOfKind<OMPFirstprivateClause>()) {
    auto IRef = C->varlist_begin();
    auto InitsRef = C->inits().begin();
    for (auto IInit : C->private_copies()) {
      auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(*IRef)->getDecl());
      bool ThisFirstprivateIsLastprivate =
          Lastprivates.count(OrigVD->getCanonicalDecl()) > 0;
      auto *CapFD = CapturesInfo.lookup(OrigVD);
      auto *FD = CapturedStmtInfo->lookup(OrigVD);
      if (!ThisFirstprivateIsLastprivate && FD && (FD == CapFD) &&
          !FD->getType()->isReferenceType()) {
        EmittedAsFirstprivate.insert(OrigVD->getCanonicalDecl());
        ++IRef;
        ++InitsRef;
        continue;
      }
      FirstprivateIsLastprivate =
          FirstprivateIsLastprivate || ThisFirstprivateIsLastprivate;
      if (EmittedAsFirstprivate.insert(OrigVD->getCanonicalDecl()).second) {
        auto *VD = cast<VarDecl>(cast<DeclRefExpr>(IInit)->getDecl());
        auto *VDInit = cast<VarDecl>(cast<DeclRefExpr>(*InitsRef)->getDecl());
        bool IsRegistered;
        DeclRefExpr DRE(const_cast<VarDecl *>(OrigVD),
                        /*RefersToEnclosingVariableOrCapture=*/FD != nullptr,
                        (*IRef)->getType(), VK_LValue, (*IRef)->getExprLoc());
        Address OriginalAddr = EmitLValue(&DRE).getAddress();
        QualType Type = VD->getType();
        if (Type->isArrayType()) {
          // Emit VarDecl with copy init for arrays.
          // Get the address of the original variable captured in current
          // captured region.
          IsRegistered = PrivateScope.addPrivate(OrigVD, [&]() -> Address {
            auto Emission = EmitAutoVarAlloca(*VD);
            auto *Init = VD->getInit();
            if (!isa<CXXConstructExpr>(Init) || isTrivialInitializer(Init)) {
              // Perform simple memcpy.
              EmitAggregateAssign(Emission.getAllocatedAddress(), OriginalAddr,
                                  Type);
            } else {
              EmitOMPAggregateAssign(
                  Emission.getAllocatedAddress(), OriginalAddr, Type,
                  [this, VDInit, Init](Address DestElement,
                                       Address SrcElement) {
                    // Clean up any temporaries needed by the initialization.
                    RunCleanupsScope InitScope(*this);
                    // Emit initialization for single element.
                    setAddrOfLocalVar(VDInit, SrcElement);
                    EmitAnyExprToMem(Init, DestElement,
                                     Init->getType().getQualifiers(),
                                     /*IsInitializer*/ false);
                    LocalDeclMap.erase(VDInit);
                  });
            }
            EmitAutoVarCleanups(Emission);
            return Emission.getAllocatedAddress();
          });
        } else {
          IsRegistered = PrivateScope.addPrivate(OrigVD, [&]() -> Address {
            // Emit private VarDecl with copy init.
            // Remap temp VDInit variable to the address of the original
            // variable
            // (for proper handling of captured global variables).
            setAddrOfLocalVar(VDInit, OriginalAddr);
            EmitDecl(*VD);
            LocalDeclMap.erase(VDInit);
            return GetAddrOfLocalVar(VD);
          });
        }
        assert(IsRegistered &&
               "firstprivate var already registered as private");
        // Silence the warning about unused variable.
        (void)IsRegistered;
      }
      ++IRef;
      ++InitsRef;
    }
  }
  return FirstprivateIsLastprivate && !EmittedAsFirstprivate.empty();
}

void CodeGenFunction::EmitOMPPrivateClause(
    const OMPExecutableDirective &D,
    CodeGenFunction::OMPPrivateScope &PrivateScope) {
  if (!HaveInsertPoint())
    return;
  llvm::DenseSet<const VarDecl *> EmittedAsPrivate;
  for (const auto *C : D.getClausesOfKind<OMPPrivateClause>()) {
    auto IRef = C->varlist_begin();
    for (auto IInit : C->private_copies()) {
      auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(*IRef)->getDecl());
      if (EmittedAsPrivate.insert(OrigVD->getCanonicalDecl()).second) {
        auto VD = cast<VarDecl>(cast<DeclRefExpr>(IInit)->getDecl());
        bool IsRegistered =
            PrivateScope.addPrivate(OrigVD, [&]() -> Address {
              // Emit private VarDecl with copy init.
              EmitDecl(*VD);
              return GetAddrOfLocalVar(VD);
            });
        assert(IsRegistered && "private var already registered as private");
        // Silence the warning about unused variable.
        (void)IsRegistered;
      }
      ++IRef;
    }
  }
}

bool CodeGenFunction::EmitOMPCopyinClause(const OMPExecutableDirective &D) {
  if (!HaveInsertPoint())
    return false;
  // threadprivate_var1 = master_threadprivate_var1;
  // operator=(threadprivate_var2, master_threadprivate_var2);
  // ...
  // __kmpc_barrier(&loc, global_tid);
  llvm::DenseSet<const VarDecl *> CopiedVars;
  llvm::BasicBlock *CopyBegin = nullptr, *CopyEnd = nullptr;
  for (const auto *C : D.getClausesOfKind<OMPCopyinClause>()) {
    auto IRef = C->varlist_begin();
    auto ISrcRef = C->source_exprs().begin();
    auto IDestRef = C->destination_exprs().begin();
    for (auto *AssignOp : C->assignment_ops()) {
      auto *VD = cast<VarDecl>(cast<DeclRefExpr>(*IRef)->getDecl());
      QualType Type = VD->getType();
      if (CopiedVars.insert(VD->getCanonicalDecl()).second) {
        // Get the address of the master variable. If we are emitting code with
        // TLS support, the address is passed from the master as field in the
        // captured declaration.
        Address MasterAddr = Address::invalid();
        if (getLangOpts().OpenMPUseTLS &&
            getContext().getTargetInfo().isTLSSupported()) {
          assert(CapturedStmtInfo->lookup(VD) &&
                 "Copyin threadprivates should have been captured!");
          DeclRefExpr DRE(const_cast<VarDecl *>(VD), true, (*IRef)->getType(),
                          VK_LValue, (*IRef)->getExprLoc());
          MasterAddr = EmitLValue(&DRE).getAddress();
          LocalDeclMap.erase(VD);
        } else {
          MasterAddr =
            Address(VD->isStaticLocal() ? CGM.getStaticLocalDeclAddress(VD)
                                        : CGM.GetAddrOfGlobal(VD),
                    getContext().getDeclAlign(VD));
        }
        // Get the address of the threadprivate variable.
        Address PrivateAddr = EmitLValue(*IRef).getAddress();
        if (CopiedVars.size() == 1) {
          // At first check if current thread is a master thread. If it is, no
          // need to copy data.
          CopyBegin = createBasicBlock("copyin.not.master");
          CopyEnd = createBasicBlock("copyin.not.master.end");
          Builder.CreateCondBr(
              Builder.CreateICmpNE(
                  Builder.CreatePtrToInt(MasterAddr.getPointer(), CGM.IntPtrTy),
                  Builder.CreatePtrToInt(PrivateAddr.getPointer(), CGM.IntPtrTy)),
              CopyBegin, CopyEnd);
          EmitBlock(CopyBegin);
        }
        auto *SrcVD = cast<VarDecl>(cast<DeclRefExpr>(*ISrcRef)->getDecl());
        auto *DestVD = cast<VarDecl>(cast<DeclRefExpr>(*IDestRef)->getDecl());
        EmitOMPCopy(Type, PrivateAddr, MasterAddr, DestVD, SrcVD, AssignOp);
      }
      ++IRef;
      ++ISrcRef;
      ++IDestRef;
    }
  }
  if (CopyEnd) {
    // Exit out of copying procedure for non-master thread.
    EmitBlock(CopyEnd, /*IsFinished=*/true);
    return true;
  }
  return false;
}

bool CodeGenFunction::EmitOMPLastprivateClauseInit(
    const OMPExecutableDirective &D, OMPPrivateScope &PrivateScope) {
  if (!HaveInsertPoint())
    return false;
  bool HasAtLeastOneLastprivate = false;
  llvm::DenseSet<const VarDecl *> SIMDLCVs;
  if (isOpenMPSimdDirective(D.getDirectiveKind())) {
    auto *LoopDirective = cast<OMPLoopDirective>(&D);
    for (auto *C : LoopDirective->counters()) {
      SIMDLCVs.insert(
          cast<VarDecl>(cast<DeclRefExpr>(C)->getDecl())->getCanonicalDecl());
    }
  }
  llvm::DenseSet<const VarDecl *> AlreadyEmittedVars;
  for (const auto *C : D.getClausesOfKind<OMPLastprivateClause>()) {
    HasAtLeastOneLastprivate = true;
    if (isOpenMPTaskLoopDirective(D.getDirectiveKind()))
      break;
    auto IRef = C->varlist_begin();
    auto IDestRef = C->destination_exprs().begin();
    for (auto *IInit : C->private_copies()) {
      // Keep the address of the original variable for future update at the end
      // of the loop.
      auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(*IRef)->getDecl());
      // Taskloops do not require additional initialization, it is done in
      // runtime support library.
      if (AlreadyEmittedVars.insert(OrigVD->getCanonicalDecl()).second) {
        auto *DestVD = cast<VarDecl>(cast<DeclRefExpr>(*IDestRef)->getDecl());
        PrivateScope.addPrivate(DestVD, [this, OrigVD, IRef]() -> Address {
          DeclRefExpr DRE(
              const_cast<VarDecl *>(OrigVD),
              /*RefersToEnclosingVariableOrCapture=*/CapturedStmtInfo->lookup(
                  OrigVD) != nullptr,
              (*IRef)->getType(), VK_LValue, (*IRef)->getExprLoc());
          return EmitLValue(&DRE).getAddress();
        });
        // Check if the variable is also a firstprivate: in this case IInit is
        // not generated. Initialization of this variable will happen in codegen
        // for 'firstprivate' clause.
        if (IInit && !SIMDLCVs.count(OrigVD->getCanonicalDecl())) {
          auto *VD = cast<VarDecl>(cast<DeclRefExpr>(IInit)->getDecl());
          bool IsRegistered = PrivateScope.addPrivate(OrigVD, [&]() -> Address {
            // Emit private VarDecl with copy init.
            EmitDecl(*VD);
            return GetAddrOfLocalVar(VD);
          });
          assert(IsRegistered &&
                 "lastprivate var already registered as private");
          (void)IsRegistered;
        }
      }
      ++IRef;
      ++IDestRef;
    }
    if (C->getModifier() == OMPC_LASTPRIVATE_conditional) {
      for (auto *CLIter : C->conditional_lastprivate_iterations()) {
        if (!CLIter)
          continue;
        llvm::APInt InitValue = llvm::APInt::getMinValue(
            getContext().getTypeSize(CLIter->getType()));
        auto Init = llvm::ConstantInt::get(getLLVMContext(), InitValue);
        EmitStoreOfScalar(Init, EmitLValue(CLIter));
      }
    }
  }
  return HasAtLeastOneLastprivate;
}

static std::pair<unsigned, unsigned>
GetLastprivateVariableCount(const OMPExecutableDirective &D) {
  unsigned UnconditionalSize = 0;
  unsigned ConditionalSize = 0;
  for (const auto *C : D.getClausesOfKind<OMPLastprivateClause>()) {
    if (C->getModifier() == OMPC_LASTPRIVATE_conditional)
      ConditionalSize +=
          std::count_if(C->conditional_lastprivate_iterations().begin(),
                        C->conditional_lastprivate_iterations().end(),
                        [](const Expr *E) { return E; });
    else
      UnconditionalSize += C->varlist_size();
  }
  return {UnconditionalSize, ConditionalSize};
}

void CodeGenFunction::EmitOMPLastprivateClauseFinal(
    const OMPExecutableDirective &D, bool NoFinals, llvm::Value *IsLastIterCond,
    bool UnconditionalKind) {
  if (!HaveInsertPoint())
    return;
  llvm::DenseSet<const VarDecl *> AlreadyEmittedVars;
  llvm::DenseMap<const VarDecl *, const Expr *> LoopCountersAndUpdates;
  if (auto *LoopDirective = dyn_cast<OMPLoopDirective>(&D)) {
    auto IC = LoopDirective->counters().begin();
    for (auto F : LoopDirective->finals()) {
      auto *D =
          cast<VarDecl>(cast<DeclRefExpr>(*IC)->getDecl())->getCanonicalDecl();
      if (NoFinals)
        AlreadyEmittedVars.insert(D);
      else
        LoopCountersAndUpdates[D] = F;
      ++IC;
    }
  }
  auto &&CodeGenCopy = [&AlreadyEmittedVars, &LoopCountersAndUpdates](
      CodeGenFunction &CGF, const Expr *AssignOp, const Expr *IRef,
      const Expr *ISrcRef, const Expr *IDestRef) {
    auto *PrivateVD = cast<VarDecl>(cast<DeclRefExpr>(IRef)->getDecl());
    QualType Type = PrivateVD->getType();
    auto *CanonicalVD = PrivateVD->getCanonicalDecl();
    // If lastprivate variable is a loop control variable for loop-based
    // directive, update its value before copyin back to original
    // variable.
    if (auto *FinalExpr = LoopCountersAndUpdates.lookup(CanonicalVD))
      CGF.EmitIgnoredExpr(FinalExpr);
    auto *SrcVD = cast<VarDecl>(cast<DeclRefExpr>(ISrcRef)->getDecl());
    auto *DestVD = cast<VarDecl>(cast<DeclRefExpr>(IDestRef)->getDecl());
    // Get the address of the original variable.
    Address OriginalAddr = CGF.GetAddrOfLocalVar(DestVD);
    // Get the address of the private variable.
    Address PrivateAddr = CGF.GetAddrOfLocalVar(PrivateVD);
    if (auto RefTy = PrivateVD->getType()->getAs<ReferenceType>())
      PrivateAddr =
          Address(CGF.Builder.CreateLoad(PrivateAddr),
                  CGF.getNaturalTypeAlignment(RefTy->getPointeeType()));
    CGF.EmitOMPCopy(Type, OriginalAddr, PrivateAddr, DestVD, SrcVD, AssignOp);
  };

  // Codegen conditional lastprivate variables.
  unsigned UnconditionalSize;
  unsigned ConditionalSize;
  std::tie(UnconditionalSize, ConditionalSize) = GetLastprivateVariableCount(D);
  if (!UnconditionalKind && ConditionalSize) {
    ASTContext &Ctx = CGM.getContext();
    llvm::APInt IdxArraySize(/*numBits=*/32, ConditionalSize);
    auto IdxArrayTy =
        Ctx.getConstantArrayType(Ctx.getSizeType(), IdxArraySize,
                                 ArrayType::Normal, /*IndexTypeQuals=*/0);
    auto IdxArray = CreateMemTemp(IdxArrayTy, "idx_array");
    unsigned I = 0;
    for (const auto *C : D.getClausesOfKind<OMPLastprivateClause>()) {
      if (C->getModifier() != OMPC_LASTPRIVATE_conditional)
        continue;
      for (auto *CLIter : C->conditional_lastprivate_iterations()) {
        if (!CLIter)
          continue;
        Address Elem =
            Builder.CreateConstArrayGEP(IdxArray, I, getPointerSize());
        auto *ITVD = cast<VarDecl>(cast<DeclRefExpr>(CLIter)->getDecl());
        Address ITAddr = GetAddrOfLocalVar(ITVD);
        Builder.CreateStore(Builder.CreateLoad(ITAddr), Elem);
        ++I;
      }
    }
    CGM.getOpenMPRuntime().emitReduceConditionalLastprivateCall(
        *this, D.getLocEnd(), Builder.getInt32(ConditionalSize),
        Builder.CreatePointerBitCastOrAddrSpaceCast(IdxArray.getPointer(),
                                                    CGM.VoidPtrTy));
    I = 0;
    for (const auto *C : D.getClausesOfKind<OMPLastprivateClause>()) {
      if (C->getModifier() != OMPC_LASTPRIVATE_conditional)
        continue;
      auto CLIter = C->conditional_lastprivate_iterations().begin();
      auto CLVar = C->conditional_lastprivate_variables().begin();
      auto IRef = C->varlist_begin();
      auto ISrcRef = C->source_exprs().begin();
      auto IDestRef = C->destination_exprs().begin();
      for (auto *AssignOp : C->assignment_ops()) {
        auto *PrivateVD = cast<VarDecl>(cast<DeclRefExpr>(*IRef)->getDecl());
        auto *CanonicalVD = PrivateVD->getCanonicalDecl();
        if (AlreadyEmittedVars.insert(CanonicalVD).second) {
          Address Elem =
              Builder.CreateConstArrayGEP(IdxArray, I, getPointerSize());
          auto MaxIdx = Builder.CreateLoad(Elem);
          auto *ITVD = cast<VarDecl>(cast<DeclRefExpr>(*CLIter)->getDecl());
          Address ITAddr = GetAddrOfLocalVar(ITVD);
          auto ThisIdx = Builder.CreateLoad(ITAddr);
          auto *IsLastUpdateCond = Builder.CreateICmpEQ(ThisIdx, MaxIdx);
          llvm::BasicBlock *ThenBB =
              createBasicBlock(".omp.conditional.lastprivate.then");
          llvm::BasicBlock *DoneBB =
              createBasicBlock(".omp.conditional.lastprivate.done");
          Builder.CreateCondBr(IsLastUpdateCond, ThenBB, DoneBB);
          EmitBlock(ThenBB);

          CodeGenCopy(*this, AssignOp, *CLVar, *ISrcRef, *IDestRef);

          EmitBlock(DoneBB, /*IsFinished=*/true);
        }

        ++CLIter;
        ++CLVar;
        ++IRef;
        ++ISrcRef;
        ++IDestRef;
        ++I;
      }
      if (auto *PostUpdate = C->getPostUpdateExpr())
        EmitIgnoredExpr(PostUpdate);
    }
  }

  if (UnconditionalKind || UnconditionalSize) {
    // Emit following code:
    // if (<IsLastIterCond>) {
    //   orig_var1 = private_orig_var1;
    //   ...
    //   orig_varn = private_orig_varn;
    // }
    llvm::BasicBlock *ThenBB = nullptr;
    llvm::BasicBlock *DoneBB = nullptr;
    if (IsLastIterCond) {
      ThenBB = createBasicBlock(".omp.lastprivate.then");
      DoneBB = createBasicBlock(".omp.lastprivate.done");
      Builder.CreateCondBr(IsLastIterCond, ThenBB, DoneBB);
      EmitBlock(ThenBB);
    }
    for (const auto *C : D.getClausesOfKind<OMPLastprivateClause>()) {
      if (!UnconditionalKind &&
          C->getModifier() == OMPC_LASTPRIVATE_conditional)
        continue;
      auto CLVar = C->conditional_lastprivate_variables().begin();
      auto IRef = C->varlist_begin();
      auto ISrcRef = C->source_exprs().begin();
      auto IDestRef = C->destination_exprs().begin();
      for (auto *AssignOp : C->assignment_ops()) {
        auto *PrivateVD = cast<VarDecl>(cast<DeclRefExpr>(*IRef)->getDecl());
        auto *CanonicalVD = PrivateVD->getCanonicalDecl();
        if (AlreadyEmittedVars.insert(CanonicalVD).second) {
          auto *Private =
              C->getModifier() == OMPC_LASTPRIVATE_conditional ? *CLVar : *IRef;
          CodeGenCopy(*this, AssignOp, Private, *ISrcRef, *IDestRef);
        }
        ++CLVar;
        ++IRef;
        ++ISrcRef;
        ++IDestRef;
      }
      if (auto *PostUpdate = C->getPostUpdateExpr())
        EmitIgnoredExpr(PostUpdate);
    }
    if (IsLastIterCond)
      EmitBlock(DoneBB, /*IsFinished=*/true);
  }
}

void CodeGenFunction::EmitOMPReductionClauseInit(
    const OMPExecutableDirective &D,
    CodeGenFunction::OMPPrivateScope &PrivateScope) {
  if (!HaveInsertPoint())
    return;
  SmallVector<const Expr *, 4> Shareds;
  SmallVector<const Expr *, 4> Privates;
  SmallVector<const Expr *, 4> ReductionOps;
  SmallVector<const Expr *, 4> LHSs;
  SmallVector<const Expr *, 4> RHSs;
  for (const auto *C : D.getClausesOfKind<OMPReductionClause>()) {
    auto IPriv = C->privates().begin();
    auto IRed = C->reduction_ops().begin();
    auto ILHS = C->lhs_exprs().begin();
    auto IRHS = C->rhs_exprs().begin();
    for (const auto *Ref : C->varlists()) {
      Shareds.emplace_back(Ref);
      Privates.emplace_back(*IPriv);
      ReductionOps.emplace_back(*IRed);
      LHSs.emplace_back(*ILHS);
      RHSs.emplace_back(*IRHS);
      std::advance(IPriv, 1);
      std::advance(IRed, 1);
      std::advance(ILHS, 1);
      std::advance(IRHS, 1);
    }
  }
  ReductionCodeGen RedCG(Shareds, Privates, ReductionOps);
  unsigned Count = 0;
  auto ILHS = LHSs.begin();
  auto IRHS = RHSs.begin();
  auto IPriv = Privates.begin();
  for (const auto *IRef : Shareds) {
    auto *PrivateVD = cast<VarDecl>(cast<DeclRefExpr>(*IPriv)->getDecl());
    // Emit private VarDecl with reduction init.
    RedCG.emitSharedLValue(*this, Count);
    RedCG.emitAggregateType(*this, Count);
    auto Emission = EmitAutoVarAlloca(*PrivateVD);
    RedCG.emitInitialization(*this, Count, Emission.getAllocatedAddress(),
                             RedCG.getSharedLValue(Count),
                             [&Emission](CodeGenFunction &CGF) {
                               CGF.EmitAutoVarInit(Emission);
                               return true;
                             });
    EmitAutoVarCleanups(Emission);
    Address BaseAddr = RedCG.adjustPrivateAddress(
        *this, Count, Emission.getAllocatedAddress());
    bool IsRegistered = PrivateScope.addPrivate(
        RedCG.getBaseDecl(Count), [BaseAddr]() -> Address { return BaseAddr; });
    assert(IsRegistered && "private var already registered as private");
    // Silence the warning about unused variable.
    (void)IsRegistered;

    auto *LHSVD = cast<VarDecl>(cast<DeclRefExpr>(*ILHS)->getDecl());
    auto *RHSVD = cast<VarDecl>(cast<DeclRefExpr>(*IRHS)->getDecl());
    QualType Type = PrivateVD->getType();
    bool isaOMPArraySectionExpr = isa<OMPArraySectionExpr>(IRef);
    if (isaOMPArraySectionExpr && Type->isVariablyModifiedType()) {
      // Store the address of the original variable associated with the LHS
      // implicit variable.
      PrivateScope.addPrivate(LHSVD, [&RedCG, Count]() -> Address {
        return RedCG.getSharedLValue(Count).getAddress();
      });
      PrivateScope.addPrivate(RHSVD, [this, PrivateVD]() -> Address {
        return GetAddrOfLocalVar(PrivateVD);
      });
    } else if ((isaOMPArraySectionExpr && Type->isScalarType()) ||
               isa<ArraySubscriptExpr>(IRef)) {
      // Store the address of the original variable associated with the LHS
      // implicit variable.
      PrivateScope.addPrivate(LHSVD, [&RedCG, Count]() -> Address {
        return RedCG.getSharedLValue(Count).getAddress();
      });
      PrivateScope.addPrivate(RHSVD, [this, PrivateVD, RHSVD]() -> Address {
        return Builder.CreateElementBitCast(GetAddrOfLocalVar(PrivateVD),
                                            ConvertTypeForMem(RHSVD->getType()),
                                            "rhs.begin");
      });
    } else {
      QualType Type = PrivateVD->getType();
      bool IsArray = getContext().getAsArrayType(Type) != nullptr;
      Address OriginalAddr = RedCG.getSharedLValue(Count).getAddress();
      // Store the address of the original variable associated with the LHS
      // implicit variable.
      if (IsArray) {
        OriginalAddr = Builder.CreateElementBitCast(
            OriginalAddr, ConvertTypeForMem(LHSVD->getType()), "lhs.begin");
      }
      PrivateScope.addPrivate(
          LHSVD, [OriginalAddr]() -> Address { return OriginalAddr; });
      PrivateScope.addPrivate(
          RHSVD, [this, PrivateVD, RHSVD, IsArray]() -> Address {
            return IsArray
                       ? Builder.CreateElementBitCast(
                             GetAddrOfLocalVar(PrivateVD),
                             ConvertTypeForMem(RHSVD->getType()), "rhs.begin")
                       : GetAddrOfLocalVar(PrivateVD);
          });
    }
    ++ILHS;
    ++IRHS;
    ++IPriv;
    ++Count;
  }
}

void CodeGenFunction::EmitOMPReductionClauseFinal(
    const OMPExecutableDirective &D, const OpenMPDirectiveKind ReductionKind) {
  if (!HaveInsertPoint())
    return;
  llvm::SmallVector<const Expr *, 8> Privates;
  llvm::SmallVector<const Expr *, 8> LHSExprs;
  llvm::SmallVector<const Expr *, 8> RHSExprs;
  llvm::SmallVector<const Expr *, 8> ReductionOps;
  bool HasAtLeastOneReduction = false;
  for (const auto *C : D.getClausesOfKind<OMPReductionClause>()) {
    HasAtLeastOneReduction = true;
    Privates.append(C->privates().begin(), C->privates().end());
    LHSExprs.append(C->lhs_exprs().begin(), C->lhs_exprs().end());
    RHSExprs.append(C->rhs_exprs().begin(), C->rhs_exprs().end());
    ReductionOps.append(C->reduction_ops().begin(), C->reduction_ops().end());
  }
  if (HasAtLeastOneReduction) {
    // Emit nowait reduction if nowait clause is present or directive is a
    // parallel directive (it always has implicit barrier).
    CGM.getOpenMPRuntime().emitReduction(
        *this, D.getLocEnd(), Privates, LHSExprs, RHSExprs, ReductionOps,
        D.getSingleClause<OMPNowaitClause>() ||
            isOpenMPParallelDirective(D.getDirectiveKind()) ||
            D.getDirectiveKind() == OMPD_simd,
        D.getDirectiveKind() == OMPD_simd, ReductionKind);
  }
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

static void emitCommonOMPParallelDirective(CodeGenFunction &CGF,
                                           const OMPExecutableDirective &S,
                                           OpenMPDirectiveKind InnermostKind,
                                           const RegionCodeGenTy &CodeGen,
                                           unsigned CaptureLevel = 1) {
  CGF.CGM.getOpenMPRuntime().registerParallelContext(CGF, S);
  const auto *CS = cast<CapturedStmt>(S.getAssociatedStmt());
  if (S.hasClausesOfKind<OMPDependClause>() &&
      isOpenMPTargetExecutionDirective(S.getDirectiveKind()))
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
  auto OutlinedFn = CGF.CGM.getOpenMPRuntime().emitParallelOutlinedFunction(
      S, *CS->getCapturedDecl()->param_begin(), InnermostKind, CodeGen,
      CaptureLevel);
  if (const auto *NumThreadsClause = S.getSingleClause<OMPNumThreadsClause>()) {
    CodeGenFunction::RunCleanupsScope NumThreadsScope(CGF);
    auto NumThreads = CGF.EmitScalarExpr(NumThreadsClause->getNumThreads(),
                                         /*IgnoreResultAssign*/ true);
    CGF.CGM.getOpenMPRuntime().emitNumThreadsClause(
        CGF, NumThreads, NumThreadsClause->getLocStart());
  }
  if (const auto *ProcBindClause = S.getSingleClause<OMPProcBindClause>()) {
    CodeGenFunction::RunCleanupsScope ProcBindScope(CGF);
    CGF.CGM.getOpenMPRuntime().emitProcBindClause(
        CGF, ProcBindClause->getProcBindKind(), ProcBindClause->getLocStart());
  }
  const Expr *IfCond = nullptr;
  for (const auto *C : S.getClausesOfKind<OMPIfClause>()) {
    if (C->getNameModifier() == OMPD_unknown ||
        C->getNameModifier() == OMPD_parallel) {
      IfCond = C->getCondition();
      break;
    }
  }

  llvm::SmallVector<llvm::Value *, 16> CapturedVars;

  OMPParallelScope Scope(CGF, S);
  // Combining 'distribute' with 'for' requires sharing each 'distribute' chunk
  // lower and upper bounds with the pragma 'for' chunking mechanism. Also, the
  // lexical scope was already initiated if parallel is not the first component
  // of a combined directive.
  if (isOpenMPLoopBoundSharingDirective(S.getDirectiveKind())) {
    const OMPLoopDirective &Dir = cast<OMPLoopDirective>(S);
    LValue LB = CGF.EmitLValue(cast<DeclRefExpr>(Dir.getLowerBoundVariable()));
    auto LBCast =
        CGF.Builder.CreateIntCast(CGF.Builder.CreateLoad(LB.getAddress()),
                                  CGF.SizeTy, /*isSigned=*/false);
    CapturedVars.push_back(LBCast);
    LValue UB = CGF.EmitLValue(cast<DeclRefExpr>(Dir.getUpperBoundVariable()));
    auto UBCast =
        CGF.Builder.CreateIntCast(CGF.Builder.CreateLoad(UB.getAddress()),
                                  CGF.SizeTy, /*isSigned=*/false);
    CapturedVars.push_back(UBCast);
  }
  CGF.GenerateOpenMPCapturedVars(*CS, CapturedVars, CaptureLevel);
  CGF.CGM.getOpenMPRuntime().emitParallelCall(CGF, S.getLocStart(), OutlinedFn,
                                              CapturedVars, IfCond);
}

void CodeGenFunction::EmitOMPParallelDirective(const OMPParallelDirective &S) {
  // Emit parallel region as a standalone region.
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    OMPPrivateScope PrivateScope(CGF);
    bool Copyins = CGF.EmitOMPCopyinClause(S);
    (void)CGF.EmitOMPFirstprivateClause(S, PrivateScope);
    if (Copyins) {
      // Emit implicit barrier to synchronize threads and avoid data races on
      // propagation master's thread values of threadprivate variables to local
      // instances of that variables of all other implicit threads.
      CGF.CGM.getOpenMPRuntime().emitBarrierCall(
          CGF, S.getLocStart(), OMPD_unknown, /*EmitChecks=*/false,
          /*ForceSimpleCall=*/true);
    }
    CGF.EmitOMPPrivateClause(S, PrivateScope);
    CGF.EmitOMPReductionClauseInit(S, PrivateScope);
    (void)PrivateScope.Privatize();
    CGF.EmitStmt(cast<CapturedStmt>(S.getAssociatedStmt())->getCapturedStmt());
    CGF.EmitOMPReductionClauseFinal(S, OMPD_parallel);
  };
  emitCommonOMPParallelDirective(*this, S, OMPD_parallel, CodeGen);
  emitPostUpdateForReductionClause(
      *this, S, [](CodeGenFunction &) -> llvm::Value * { return nullptr; });
}

void CodeGenFunction::EmitOMPLoopBody(const OMPLoopDirective &D,
                                      JumpDest LoopExit) {
  RunCleanupsScope BodyScope(*this);
  // Update counters values on current iteration.
  for (auto I : D.updates()) {
    EmitIgnoredExpr(I);
  }
  // Update the linear variables.
  for (const auto *C : D.getClausesOfKind<OMPLinearClause>()) {
    for (auto *U : C->updates())
      EmitIgnoredExpr(U);
  }

  // On a continue in the body, jump to the end.
  auto Continue = getJumpDestInCurrentScope("omp.body.continue");
  BreakContinueStack.push_back(BreakContinue(LoopExit, Continue));
  // Emit loop body.
  EmitStmt(D.getBody());
  // The end (updates/cleanups).
  EmitBlock(Continue.getBlock());
  BreakContinueStack.pop_back();
}

void CodeGenFunction::EmitOMPInnerLoop(
    const Stmt &S, bool RequiresCleanup, const Expr *LoopCond,
    const Expr *IncExpr, const Expr *LastprivateIterInitExpr,
    const llvm::function_ref<void(CodeGenFunction &)> &BodyGen,
    const llvm::function_ref<void(CodeGenFunction &)> &PostIncGen) {
  auto LoopExit = getJumpDestInCurrentScope("omp.inner.for.end");

  // Start the loop with a block that tests the condition.
  auto CondBlock = createBasicBlock("omp.inner.for.cond");
  EmitBlock(CondBlock);
  const SourceRange &R = S.getSourceRange();
  LoopStack.push(CondBlock, SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));

  // If there are any cleanups between here and the loop-exit scope,
  // create a block to stage a loop exit along.
  auto ExitBlock = LoopExit.getBlock();
  if (RequiresCleanup)
    ExitBlock = createBasicBlock("omp.inner.for.cond.cleanup");

  auto LoopBody = createBasicBlock("omp.inner.for.body");

  // Emit condition.
  EmitBranchOnBoolExpr(LoopCond, LoopBody, ExitBlock, getProfileCount(&S));
  if (ExitBlock != LoopExit.getBlock()) {
    EmitBlock(ExitBlock);
    EmitBranchThroughCleanup(LoopExit);
  }

  EmitBlock(LoopBody);
  incrementProfileCounter(&S);

  // Create a block for the increment.
  auto Continue = getJumpDestInCurrentScope("omp.inner.for.inc");
  BreakContinueStack.push_back(BreakContinue(LoopExit, Continue));

  if (LastprivateIterInitExpr)
    EmitIgnoredExpr(LastprivateIterInitExpr);

  BodyGen(*this);

  // Emit "IV = IV + 1" and a back-edge to the condition block.
  EmitBlock(Continue.getBlock());
  EmitIgnoredExpr(IncExpr);
  PostIncGen(*this);
  BreakContinueStack.pop_back();
  EmitBranch(CondBlock);
  LoopStack.pop();
  // Emit the fall-through block.
  EmitBlock(LoopExit.getBlock());
}

bool CodeGenFunction::EmitOMPLinearClauseInit(const OMPLoopDirective &D) {
  if (!HaveInsertPoint())
    return false;
  // Emit inits for the linear variables.
  bool HasLinears = false;
  for (const auto *C : D.getClausesOfKind<OMPLinearClause>()) {
    for (auto *Init : C->inits()) {
      HasLinears = true;
      auto *VD = cast<VarDecl>(cast<DeclRefExpr>(Init)->getDecl());
      if (auto *Ref = dyn_cast<DeclRefExpr>(VD->getInit()->IgnoreImpCasts())) {
        AutoVarEmission Emission = EmitAutoVarAlloca(*VD);
        auto *OrigVD = cast<VarDecl>(Ref->getDecl());
        DeclRefExpr DRE(const_cast<VarDecl *>(OrigVD),
                        CapturedStmtInfo->lookup(OrigVD) != nullptr,
                        VD->getInit()->getType(), VK_LValue,
                        VD->getInit()->getExprLoc());
        EmitExprAsInit(&DRE, VD, MakeAddrLValue(Emission.getAllocatedAddress(),
                                                VD->getType()),
                       /*capturedByInit=*/false);
        EmitAutoVarCleanups(Emission);
      } else
        EmitVarDecl(*VD);
    }
    // Emit the linear steps for the linear clauses.
    // If a step is not constant, it is pre-calculated before the loop.
    if (auto CS = cast_or_null<BinaryOperator>(C->getCalcStep()))
      if (auto SaveRef = cast<DeclRefExpr>(CS->getLHS())) {
        EmitVarDecl(*cast<VarDecl>(SaveRef->getDecl()));
        // Emit calculation of the linear step.
        EmitIgnoredExpr(CS);
      }
  }
  return HasLinears;
}

void CodeGenFunction::EmitOMPLinearClauseFinal(
    const OMPLoopDirective &D,
    const llvm::function_ref<llvm::Value *(CodeGenFunction &)> &CondGen) {
  if (!HaveInsertPoint())
    return;
  llvm::BasicBlock *DoneBB = nullptr;
  // Emit the final values of the linear variables.
  for (const auto *C : D.getClausesOfKind<OMPLinearClause>()) {
    auto IC = C->varlist_begin();
    for (auto *F : C->finals()) {
      if (!DoneBB) {
        if (auto *Cond = CondGen(*this)) {
          // If the first post-update expression is found, emit conditional
          // block if it was requested.
          auto *ThenBB = createBasicBlock(".omp.linear.pu");
          DoneBB = createBasicBlock(".omp.linear.pu.done");
          Builder.CreateCondBr(Cond, ThenBB, DoneBB);
          EmitBlock(ThenBB);
        }
      }
      auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(*IC)->getDecl());
      DeclRefExpr DRE(const_cast<VarDecl *>(OrigVD),
                      CapturedStmtInfo->lookup(OrigVD) != nullptr,
                      (*IC)->getType(), VK_LValue, (*IC)->getExprLoc());
      Address OrigAddr = EmitLValue(&DRE).getAddress();
      CodeGenFunction::OMPPrivateScope VarScope(*this);
      VarScope.addPrivate(OrigVD, [OrigAddr]() -> Address { return OrigAddr; });
      (void)VarScope.Privatize();
      EmitIgnoredExpr(F);
      ++IC;
    }
    if (auto *PostUpdate = C->getPostUpdateExpr())
      EmitIgnoredExpr(PostUpdate);
  }
  if (DoneBB)
    EmitBlock(DoneBB, /*IsFinished=*/true);
}

static void emitAlignedClause(CodeGenFunction &CGF,
                              const OMPExecutableDirective &D) {
  if (!CGF.HaveInsertPoint())
    return;
  for (const auto *Clause : D.getClausesOfKind<OMPAlignedClause>()) {
    unsigned ClauseAlignment = 0;
    if (auto AlignmentExpr = Clause->getAlignment()) {
      auto AlignmentCI =
          cast<llvm::ConstantInt>(CGF.EmitScalarExpr(AlignmentExpr));
      ClauseAlignment = static_cast<unsigned>(AlignmentCI->getZExtValue());
    }
    for (auto E : Clause->varlists()) {
      unsigned Alignment = ClauseAlignment;
      if (Alignment == 0) {
        // OpenMP [2.8.1, Description]
        // If no optional parameter is specified, implementation-defined default
        // alignments for SIMD instructions on the target platforms are assumed.
        Alignment =
            CGF.getContext()
                .toCharUnitsFromBits(CGF.getContext().getOpenMPDefaultSimdAlign(
                    E->getType()->getPointeeType()))
                .getQuantity();
      }
      assert((Alignment == 0 || llvm::isPowerOf2_32(Alignment)) &&
             "alignment is not power of 2");
      if (Alignment != 0) {
        llvm::Value *PtrValue = CGF.EmitScalarExpr(E);
        CGF.EmitAlignmentAssumption(PtrValue, Alignment);
      }
    }
  }
}

void CodeGenFunction::EmitOMPPrivateLoopCounters(
    const OMPLoopDirective &S, CodeGenFunction::OMPPrivateScope &LoopScope) {
  if (!HaveInsertPoint())
    return;
  auto I = S.private_counters().begin();
  for (auto *E : S.counters()) {
    auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    auto *PrivateVD = cast<VarDecl>(cast<DeclRefExpr>(*I)->getDecl());
    (void)LoopScope.addPrivate(VD, [&]() -> Address {
      // Emit var without initialization.
      if (!LocalDeclMap.count(PrivateVD)) {
        auto VarEmission = EmitAutoVarAlloca(*PrivateVD);
        EmitAutoVarCleanups(VarEmission);
      }
      DeclRefExpr DRE(const_cast<VarDecl *>(PrivateVD),
                      /*RefersToEnclosingVariableOrCapture=*/false,
                      (*I)->getType(), VK_LValue, (*I)->getExprLoc());
      return EmitLValue(&DRE).getAddress();
    });
    if (LocalDeclMap.count(VD) || CapturedStmtInfo->lookup(VD) ||
        VD->hasGlobalStorage()) {
      (void)LoopScope.addPrivate(PrivateVD, [&]() -> Address {
        DeclRefExpr DRE(const_cast<VarDecl *>(VD),
                        LocalDeclMap.count(VD) || CapturedStmtInfo->lookup(VD),
                        E->getType(), VK_LValue, E->getExprLoc());
        return EmitLValue(&DRE).getAddress();
      });
    }
    ++I;
  }
}

static void emitPreCond(CodeGenFunction &CGF, const OMPLoopDirective &S,
                        const Expr *Cond, llvm::BasicBlock *TrueBlock,
                        llvm::BasicBlock *FalseBlock, uint64_t TrueCount) {
  if (!CGF.HaveInsertPoint())
    return;
  {
    CodeGenFunction::OMPPrivateScope PreCondScope(CGF);
    CGF.EmitOMPPrivateLoopCounters(S, PreCondScope);
    (void)PreCondScope.Privatize();
    // Get initial values of real counters.
    for (auto I : S.inits()) {
      CGF.EmitIgnoredExpr(I);
    }
  }
  // Check that loop is executed at least one time.
  CGF.EmitBranchOnBoolExpr(Cond, TrueBlock, FalseBlock, TrueCount);
}

void CodeGenFunction::EmitOMPLinearClause(
    const OMPLoopDirective &D, CodeGenFunction::OMPPrivateScope &PrivateScope) {
  if (!HaveInsertPoint())
    return;
  llvm::DenseSet<const VarDecl *> SIMDLCVs;
  if (isOpenMPSimdDirective(D.getDirectiveKind())) {
    auto *LoopDirective = cast<OMPLoopDirective>(&D);
    for (auto *C : LoopDirective->counters()) {
      SIMDLCVs.insert(
          cast<VarDecl>(cast<DeclRefExpr>(C)->getDecl())->getCanonicalDecl());
    }
  }
  for (const auto *C : D.getClausesOfKind<OMPLinearClause>()) {
    auto CurPrivate = C->privates().begin();
    for (auto *E : C->varlists()) {
      auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
      auto *PrivateVD =
          cast<VarDecl>(cast<DeclRefExpr>(*CurPrivate)->getDecl());
      if (!SIMDLCVs.count(VD->getCanonicalDecl())) {
        bool IsRegistered = PrivateScope.addPrivate(VD, [&]() -> Address {
          // Emit private VarDecl with copy init.
          EmitVarDecl(*PrivateVD);
          return GetAddrOfLocalVar(PrivateVD);
        });
        assert(IsRegistered && "linear var already registered as private");
        // Silence the warning about unused variable.
        (void)IsRegistered;
      } else
        EmitVarDecl(*PrivateVD);
      ++CurPrivate;
    }
  }
}

static void emitSimdlenSafelenClause(CodeGenFunction &CGF,
                                     const OMPExecutableDirective &D,
                                     bool IsMonotonic) {
  if (!CGF.HaveInsertPoint())
    return;
  if (const auto *C = D.getSingleClause<OMPSimdlenClause>()) {
    RValue Len = CGF.EmitAnyExpr(C->getSimdlen(), AggValueSlot::ignored(),
                                 /*ignoreResult=*/true);
    llvm::ConstantInt *Val = cast<llvm::ConstantInt>(Len.getScalarVal());
    CGF.LoopStack.setVectorizeWidth(Val->getZExtValue());
    // In presence of finite 'safelen', it may be unsafe to mark all
    // the memory instructions parallel, because loop-carried
    // dependences of 'safelen' iterations are possible.
    if (!IsMonotonic)
      CGF.LoopStack.setParallel(!D.getSingleClause<OMPSafelenClause>());
  } else if (const auto *C = D.getSingleClause<OMPSafelenClause>()) {
    RValue Len = CGF.EmitAnyExpr(C->getSafelen(), AggValueSlot::ignored(),
                                 /*ignoreResult=*/true);
    llvm::ConstantInt *Val = cast<llvm::ConstantInt>(Len.getScalarVal());
    CGF.LoopStack.setVectorizeWidth(Val->getZExtValue());
    // In presence of finite 'safelen', it may be unsafe to mark all
    // the memory instructions parallel, because loop-carried
    // dependences of 'safelen' iterations are possible.
    CGF.LoopStack.setParallel(false);
  }
}

void CodeGenFunction::EmitOMPSimdInit(const OMPLoopDirective &D,
                                      bool IsMonotonic) {
  // Walk clauses and process safelen/lastprivate.
  LoopStack.setParallel(!IsMonotonic);
  LoopStack.setVectorizeEnable(true);
  emitSimdlenSafelenClause(*this, D, IsMonotonic);
}

void CodeGenFunction::EmitOMPSimdFinal(
    const OMPLoopDirective &D,
    const llvm::function_ref<llvm::Value *(CodeGenFunction &)> &CondGen) {
  if (!HaveInsertPoint())
    return;
  llvm::BasicBlock *DoneBB = nullptr;
  auto IC = D.counters().begin();
  auto IPC = D.private_counters().begin();
  for (auto F : D.finals()) {
    auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>((*IC))->getDecl());
    auto *PrivateVD = cast<VarDecl>(cast<DeclRefExpr>((*IPC))->getDecl());
    auto *CED = dyn_cast<OMPCapturedExprDecl>(OrigVD);
    if (LocalDeclMap.count(OrigVD) || CapturedStmtInfo->lookup(OrigVD) ||
        OrigVD->hasGlobalStorage() || CED) {
      if (!DoneBB) {
        if (auto *Cond = CondGen(*this)) {
          // If the first post-update expression is found, emit conditional
          // block if it was requested.
          auto *ThenBB = createBasicBlock(".omp.final.then");
          DoneBB = createBasicBlock(".omp.final.done");
          Builder.CreateCondBr(Cond, ThenBB, DoneBB);
          EmitBlock(ThenBB);
        }
      }
      Address OrigAddr = Address::invalid();
      if (CED)
        OrigAddr = EmitLValue(CED->getInit()->IgnoreImpCasts()).getAddress();
      else {
        DeclRefExpr DRE(const_cast<VarDecl *>(PrivateVD),
                        /*RefersToEnclosingVariableOrCapture=*/false,
                        (*IPC)->getType(), VK_LValue, (*IPC)->getExprLoc());
        OrigAddr = EmitLValue(&DRE).getAddress();
      }
      OMPPrivateScope VarScope(*this);
      VarScope.addPrivate(OrigVD,
                          [OrigAddr]() -> Address { return OrigAddr; });
      (void)VarScope.Privatize();
      EmitIgnoredExpr(F);
    }
    ++IC;
    ++IPC;
  }
  if (DoneBB)
    EmitBlock(DoneBB, /*IsFinished=*/true);
}

/// \brief Emit a helper variable and return corresponding lvalue.
LValue CodeGenFunction::EmitOMPHelperVar(const DeclRefExpr *Helper) {
  auto *VDecl = cast<VarDecl>(Helper->getDecl());
  EmitOMPHelperVar(VDecl);
  return EmitLValue(Helper);
}
void CodeGenFunction::EmitOMPHelperVar(const VarDecl *VDecl) {
  // Don't need to emit variable if it was emitted before.
  if (LocalDeclMap.find(VDecl) == LocalDeclMap.end())
    EmitVarDecl(*VDecl);
}

static void emitDeviceOMPSimdDirective(CodeGenFunction &CGF,
                                       const OMPExecutableDirective &S,
                                       OpenMPDirectiveKind InnermostKind,
                                       const RegionCodeGenTy &CodeGen) {
  CGF.CGM.getOpenMPRuntime().registerParallelContext(CGF, S);
  const auto *CS = cast<CapturedStmt>(S.getAssociatedStmt());
  if (S.hasClausesOfKind<OMPDependClause>() &&
      isOpenMPTargetExecutionDirective(S.getDirectiveKind()))
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
  llvm::SmallVector<llvm::Value *, 16> CapturedVars;

  // Capture any implicit OpenMP variables
  CGF.GenerateOpenMPCapturedVars(*CS, CapturedVars);
  auto *LaneId = CS->getCapturedDecl()->param_begin();
  auto *NumLanes = std::next(LaneId);
  auto OutlinedFn = CGF.CGM.getOpenMPRuntime().emitSimdOutlinedFunction(
      S, *LaneId, *NumLanes, InnermostKind, CodeGen);
  if (const auto *C = S.getSingleClause<OMPSimdlenClause>()) {
    RValue Len = CGF.EmitAnyExpr(C->getSimdlen(), AggValueSlot::ignored(),
                                 /*ignoreResult=*/true);
    llvm::ConstantInt *Val = cast<llvm::ConstantInt>(Len.getScalarVal());
    CodeGenFunction::RunCleanupsScope SimdLimitScope(CGF);
    CGF.CGM.getOpenMPRuntime().emitSimdLimit(CGF, Val, C->getLocStart());
  } else if (const auto *C = S.getSingleClause<OMPSafelenClause>()) {
    RValue Len = CGF.EmitAnyExpr(C->getSafelen(), AggValueSlot::ignored(),
                                 /*ignoreResult=*/true);
    llvm::ConstantInt *Val = cast<llvm::ConstantInt>(Len.getScalarVal());
    CodeGenFunction::RunCleanupsScope SimdLimitScope(CGF);
    CGF.CGM.getOpenMPRuntime().emitSimdLimit(CGF, Val, C->getLocStart());
  }
  CGF.CGM.getOpenMPRuntime().emitSimdCall(CGF, S.getLocStart(), OutlinedFn,
                                          CapturedVars);
}

void CodeGenFunction::EmitOMPSimdLoop(const OMPLoopDirective &S,
                                      bool OutlinedSimd) {
  auto &&CodeGen = [&S, OutlinedSimd](CodeGenFunction &CGF, PrePostActionTy &) {
    // Do not emit this code if it was already emitted before
    // in the same scope.
    OMPLoopScope PreInitScope(
        CGF, S, !requiresAdditionalIterationVar(S.getDirectiveKind()));
    // if (PreCond) {
    //   for (IV in 0..LastIteration) BODY;
    //   <Final counter/linear vars updates>;
    // }
    //

    // Emit: if (PreCond) - begin.
    // If the condition constant folds and can be elided, avoid emitting the
    // whole loop.
    bool CondConstant;
    llvm::BasicBlock *ContBlock = nullptr;
    if (CGF.ConstantFoldsToSimpleInteger(S.getPreCond(), CondConstant)) {
      if (!CondConstant)
        return;
    } else {
      auto *ThenBlock = CGF.createBasicBlock("simd.if.then");
      ContBlock = CGF.createBasicBlock("simd.if.end");
      emitPreCond(CGF, S, S.getPreCond(), ThenBlock, ContBlock,
                  CGF.getProfileCount(&S));
      CGF.EmitBlock(ThenBlock);
      CGF.incrementProfileCounter(&S);
    }

    // Emit the lane init and num lanes variables for the nvptx device.
    if (OutlinedSimd) {
      const Expr *LIExpr = S.getLaneInit();
      const VarDecl *LIDecl =
          cast<VarDecl>(cast<DeclRefExpr>(LIExpr)->getDecl());
      CGF.EmitVarDecl(*LIDecl);
      LValue LI = CGF.EmitLValue(LIExpr);
      CGF.EmitStoreOfScalar(
          CGF.CGM.getOpenMPRuntime().getLaneID(CGF, S.getLocStart()), LI);

      const Expr *NLExpr = S.getNumLanes();
      const VarDecl *NLDecl =
          cast<VarDecl>(cast<DeclRefExpr>(NLExpr)->getDecl());
      CGF.EmitVarDecl(*NLDecl);
      LValue NL = CGF.EmitLValue(NLExpr);
      CGF.EmitStoreOfScalar(
          CGF.CGM.getOpenMPRuntime().getNumLanes(CGF, S.getLocStart()), NL);
    }

    // Emit the loop iteration variable.
    const Expr *IVExpr =
        requiresAdditionalIterationVar(S.getDirectiveKind()) && !OutlinedSimd
            ? S.getInnermostIterationVariable()
            : S.getIterationVariable();
    const VarDecl *IVDecl = cast<VarDecl>(cast<DeclRefExpr>(IVExpr)->getDecl());
    CGF.EmitVarDecl(*IVDecl);

    // Emit Iteration variable initialization
    CGF.EmitIgnoredExpr(S.getInit());

    // Emit the conditional lastprivate iteration variable, if required.
    if (S.getConditionalLastprivateIterVariable()) {
      auto CLIVExpr =
          dyn_cast<DeclRefExpr>(S.getConditionalLastprivateIterVariable());
      auto CLIVDecl = cast<VarDecl>(CLIVExpr->getDecl());
      CGF.EmitVarDecl(*CLIVDecl);
    }

    // Emit the iterations count variable.
    // If it is not a variable, Sema decided to calculate iterations count on
    // each iteration (e.g., it is foldable into a constant).
    if (auto LIExpr = dyn_cast<DeclRefExpr>(S.getLastIteration())) {
      CGF.EmitVarDecl(*cast<VarDecl>(LIExpr->getDecl()));
      // Emit calculation of the iterations count.
      CGF.EmitIgnoredExpr(S.getCalcLastIteration());
    }

    CGF.EmitOMPSimdInit(S);

    emitAlignedClause(CGF, S);
    (void)CGF.EmitOMPLinearClauseInit(S);
    {
      OMPPrivateScope LoopScope(CGF);
      CGF.EmitOMPPrivateLoopCounters(S, LoopScope);
      CGF.EmitOMPLinearClause(S, LoopScope);
      bool LastprivateAlreadyEmitted =
          requiresAdditionalIterationVar(S.getDirectiveKind()) ||
          S.getDirectiveKind() == OMPD_target_simd;
      if (!LastprivateAlreadyEmitted)
        CGF.EmitOMPPrivateClause(S, LoopScope);
      CGF.EmitOMPReductionClauseInit(S, LoopScope);
      bool HasLastprivateClause = false;
      if (!LastprivateAlreadyEmitted)
        HasLastprivateClause = CGF.EmitOMPLastprivateClauseInit(S, LoopScope);
      (void)LoopScope.Privatize();
      CGF.EmitOMPInnerLoop(S, LoopScope.requiresCleanups(), S.getCond(),
                           S.getInc(), S.getConditionalLastprivateIterInit(),
                           [&S](CodeGenFunction &CGF) {
                             CGF.EmitOMPLoopBody(S, JumpDest());
                             CGF.EmitStopPoint(&S);
                           },
                           [](CodeGenFunction &) {});
      CGF.EmitOMPSimdFinal(
          S, [](CodeGenFunction &) -> llvm::Value * { return nullptr; });
      // Emit final copy of the lastprivate variables at the end of loops.
      if (HasLastprivateClause)
        CGF.EmitOMPLastprivateClauseFinal(S, /*NoFinals=*/true,
                                          /*IsLastIter=*/nullptr,
                                          /*UnconditionalKind=*/true);
      CGF.EmitOMPReductionClauseFinal(S, OMPD_simd);
      emitPostUpdateForReductionClause(
          CGF, S, [](CodeGenFunction &) -> llvm::Value * { return nullptr; });
    }
    CGF.EmitOMPLinearClauseFinal(
        S, [](CodeGenFunction &) -> llvm::Value * { return nullptr; });
    // Emit: if (PreCond) - end.
    if (ContBlock) {
      CGF.EmitBranch(ContBlock);
      CGF.EmitBlock(ContBlock, true);
    }
  };

  if (OutlinedSimd)
    emitDeviceOMPSimdDirective(*this, S, OMPD_simd, CodeGen);
  else {
    if (S.getDirectiveKind() == OMPD_teams_distribute_simd ||
        S.getDirectiveKind() == OMPD_target_teams_distribute_simd) {
      CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_simd, CodeGen);
    } else {
      OMPLexicalScope Scope(*this, S, /*AsInlined=*/true);
      CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_simd, CodeGen);
    }
  }
}

void CodeGenFunction::EmitOMPSimdDirective(const OMPSimdDirective &S) {
  bool OutlinedSimd =
      CGM.getLangOpts().OpenMPIsDevice && CGM.getTriple().isNVPTX();
  EmitOMPSimdLoop(S, OutlinedSimd);
}

void CodeGenFunction::EmitOMPOuterLoop(
    bool DynamicOrOrdered, bool IsMonotonic, bool IsDistribute,
    const OMPLoopDirective &S, OMPPrivateScope &LoopScope, bool Ordered,
    Address LB, Address UB, Address ST, Address IL, llvm::Value *Chunk,
    const RegionCodeGenTy &CodeGenDistributeLoopContent) {
  auto &RT = CGM.getOpenMPRuntime();

  const Expr *IVExpr = S.getIterationVariable();
  const unsigned IVSize = getContext().getTypeSize(IVExpr->getType());
  const bool IVSigned = IVExpr->getType()->hasSignedIntegerRepresentation();

  auto LoopExit = getJumpDestInCurrentScope("omp.dispatch.end");

  // Start the loop with a block that tests the condition.
  auto CondBlock = createBasicBlock("omp.dispatch.cond");
  EmitBlock(CondBlock);
  const SourceRange &R = S.getSourceRange();
  LoopStack.push(CondBlock, SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));

  llvm::Value *BoolCondVal = nullptr;
  if (!DynamicOrOrdered) {
    // UB = min(UB, GlobalUB)
    // when dealing with composite distribute parallel for and when implementing
    // the #for part: use the
    // distribute UB instead, as it is not incremented and guaranteed to be
    // less than GlobalUB
    Expr *EUB = (isOpenMPLoopBoundSharingDirective(S.getDirectiveKind()) &&
                 !IsDistribute)
                    ? S.getPrevEnsureUpperBound()
                    : S.getEnsureUpperBound();
    EmitIgnoredExpr(EUB); // S.getEnsureUpperBound());
    // IV = LB
    EmitIgnoredExpr(S.getInit());
    // IV < UB
    BoolCondVal = EvaluateExprAsBool(
        (isOpenMPLoopBoundSharingDirective(S.getDirectiveKind()) &&
         IsDistribute)
            ? S.getDistCond()
            : S.getCond());
  } else {
    BoolCondVal = RT.emitForNext(*this, S.getLocStart(), IVSize, IVSigned, IL,
                                 LB, UB, ST);
    if (isOpenMPLoopBoundSharingDirective(S.getDirectiveKind()) &&
        CGM.getLangOpts().OpenMPIsDevice && CGM.getTriple().isNVPTX()) {
      // De-normalize loop.
      QualType IteratorTy = IVExpr->getType();

      // Get the values of the current upper and power bounds.
      llvm::Value *LBVal =
          EmitLoadOfScalar(LB, /*Volatile=*/false, IteratorTy, S.getLocStart());
      llvm::Value *UBVal =
          EmitLoadOfScalar(UB, /*Volatile=*/false, IteratorTy, S.getLocStart());

      // Get the values of the previous lower bound and make sure it is casted
      // to the same type as the loop iterator variable.
      auto PrevLB = EmitLValue(S.getPrevLowerBoundVariable());
      llvm::Value *PrevLBVal = EmitLoadOfScalar(PrevLB, S.getLocStart());
      PrevLBVal = Builder.CreateIntCast(PrevLBVal, Builder.getIntNTy(IVSize),
                                        /*IsSigned=*/false);
      EmitStoreOfScalar(Builder.CreateAdd(LBVal, PrevLBVal), LB,
                        /*Volatile=*/false, IteratorTy);
      EmitStoreOfScalar(Builder.CreateAdd(UBVal, PrevLBVal), UB,
                        /*Volatile=*/false, IteratorTy);
    }
  }

  // If there are any cleanups between here and the loop-exit scope,
  // create a block to stage a loop exit along.
  auto ExitBlock = LoopExit.getBlock();
  if (LoopScope.requiresCleanups())
    ExitBlock = createBasicBlock("omp.dispatch.cleanup");

  auto LoopBody = createBasicBlock("omp.dispatch.body");
  Builder.CreateCondBr(BoolCondVal, LoopBody, ExitBlock);
  if (ExitBlock != LoopExit.getBlock()) {
    EmitBlock(ExitBlock);
    EmitBranchThroughCleanup(LoopExit);
  }
  EmitBlock(LoopBody);

  // Emit "IV = LB" (in case of static schedule, we have already calculated new
  // LB for loop condition and emitted it above).
  if (DynamicOrOrdered)
    EmitIgnoredExpr(S.getInit());

  // Create a block for the increment.
  auto Continue = getJumpDestInCurrentScope("omp.dispatch.inc");
  BreakContinueStack.push_back(BreakContinue(LoopExit, Continue));

  // Generate !llvm.loop.parallel metadata for loads and stores for loops
  // with dynamic/guided scheduling and without ordered clause.
  if (!isOpenMPSimdDirective(S.getDirectiveKind()))
    LoopStack.setParallel(!IsMonotonic);
  else
    EmitOMPSimdInit(S, IsMonotonic);

  SourceLocation Loc = S.getLocStart();
  EmitOMPInnerLoop(
      S, LoopScope.requiresCleanups(),
      (isOpenMPLoopBoundSharingDirective(S.getDirectiveKind()) && IsDistribute)
          ? S.getDistCond()
          : S.getCond(),
      (isOpenMPLoopBoundSharingDirective(S.getDirectiveKind()) && IsDistribute)
          ? S.getDistInc()
          : S.getInc(),
      S.getConditionalLastprivateIterInit(),
      [&S, IsDistribute, &CodeGenDistributeLoopContent,
       LoopExit](CodeGenFunction &CGF) {
        if (isOpenMPLoopBoundSharingDirective(S.getDirectiveKind()) &&
            IsDistribute) {
          CodeGenDistributeLoopContent(CGF);
        } else {
          CGF.EmitOMPLoopBody(S, LoopExit);
          CGF.EmitStopPoint(&S);
        }
      },
      [Ordered, IVSize, IVSigned, Loc](CodeGenFunction &CGF) {
        if (Ordered) {
          CGF.CGM.getOpenMPRuntime().emitForOrderedIterationEnd(
              CGF, Loc, IVSize, IVSigned);
        }
      });

  EmitBlock(Continue.getBlock());
  BreakContinueStack.pop_back();
  if (!DynamicOrOrdered) {
    // Emit "LB = LB + Stride", "UB = UB + Stride".
    EmitIgnoredExpr(S.getNextLowerBound());
    EmitIgnoredExpr(S.getNextUpperBound());
  }

  EmitBranch(CondBlock);
  LoopStack.pop();
  // Emit the fall-through block.
  EmitBlock(LoopExit.getBlock());

  // Tell the runtime we are done.
  auto &&CodeGen = [DynamicOrOrdered, &S, IVSize,
                    IVSigned](CodeGenFunction &CGF) {
    if (DynamicOrOrdered)
      CGF.CGM.getOpenMPRuntime().emitForDispatchFinish(CGF, S, S.getLocEnd(),
                                                       IVSize, IVSigned);
    else
      CGF.CGM.getOpenMPRuntime().emitForStaticFinish(CGF, S.getLocEnd(),
                                                     S.getDirectiveKind());
  };
  OMPCancelStack.emitExit(*this, S.getDirectiveKind(), CodeGen);
}

void CodeGenFunction::EmitOMPForOuterLoop(
    const OpenMPScheduleTy &ScheduleKind, bool IsMonotonic,
    const OMPLoopDirective &S, OMPPrivateScope &LoopScope, bool Ordered,
    Address LB, Address UB, Address ST, Address IL, llvm::Value *Chunk) {
  auto &RT = CGM.getOpenMPRuntime();

  // Dynamic scheduling of the outer loop (dynamic, guided, auto, runtime).
  const bool DynamicOrOrdered =
      Ordered || RT.isDynamic(ScheduleKind.Schedule);

  assert((Ordered ||
          !RT.isStaticNonchunked(ScheduleKind.Schedule,
                                 /*Chunked=*/Chunk != nullptr)) &&
         "static non-chunked schedule does not need outer loop");

  // Emit outer loop.
  //
  // OpenMP [2.7.1, Loop Construct, Description, table 2-1]
  // When schedule(dynamic,chunk_size) is specified, the iterations are
  // distributed to threads in the team in chunks as the threads request them.
  // Each thread executes a chunk of iterations, then requests another chunk,
  // until no chunks remain to be distributed. Each chunk contains chunk_size
  // iterations, except for the last chunk to be distributed, which may have
  // fewer iterations. When no chunk_size is specified, it defaults to 1.
  //
  // When schedule(guided,chunk_size) is specified, the iterations are assigned
  // to threads in the team in chunks as the executing threads request them.
  // Each thread executes a chunk of iterations, then requests another chunk,
  // until no chunks remain to be assigned. For a chunk_size of 1, the size of
  // each chunk is proportional to the number of unassigned iterations divided
  // by the number of threads in the team, decreasing to 1. For a chunk_size
  // with value k (greater than 1), the size of each chunk is determined in the
  // same way, with the restriction that the chunks do not contain fewer than k
  // iterations (except for the last chunk to be assigned, which may have fewer
  // than k iterations).
  //
  // When schedule(auto) is specified, the decision regarding scheduling is
  // delegated to the compiler and/or runtime system. The programmer gives the
  // implementation the freedom to choose any possible mapping of iterations to
  // threads in the team.
  //
  // When schedule(runtime) is specified, the decision regarding scheduling is
  // deferred until run time, and the schedule and chunk size are taken from the
  // run-sched-var ICV. If the ICV is set to auto, the schedule is
  // implementation defined
  //
  // while(__kmpc_dispatch_next(&LB, &UB)) {
  //   idx = LB;
  //   while (idx <= UB) { BODY; ++idx;
  //   __kmpc_dispatch_fini_(4|8)[u](); // For ordered loops only.
  //   } // inner loop
  // }
  //
  // OpenMP [2.7.1, Loop Construct, Description, table 2-1]
  // When schedule(static, chunk_size) is specified, iterations are divided into
  // chunks of size chunk_size, and the chunks are assigned to the threads in
  // the team in a round-robin fashion in the order of the thread number.
  //
  // while(UB = min(UB, GlobalUB), idx = LB, idx < UB) {
  //   while (idx <= UB) { BODY; ++idx; } // inner loop
  //   LB = LB + ST;
  //   UB = UB + ST;
  // }
  //
  // NVPTX: When pragma for is found composed or combined with distribute,
  // the loop bounds are not normalized. Codegen the following
  // (init: LB == DIST_LB)
  // LB = 0;
  // UB = UB - DIST_LB;
  // kmpc_dispatch_init(.., LB, UB, ..)
  // while (!kmpc_dispatch_next(.., &LB, &UB, ..) {
  //   LB += DIST_LB;
  //   UB += DIST_LB;
  //   idx = LB;
  //   while (idx <= UB) { BODY; ++idx; } // inner loop
  // }
  // kmpc_dispatch_fini();

  const Expr *IVExpr = S.getIterationVariable();
  QualType IteratorTy = IVExpr->getType();
  const unsigned IVSize = getContext().getTypeSize(IteratorTy);
  const bool IVSigned = IVExpr->getType()->hasSignedIntegerRepresentation();

  if (DynamicOrOrdered) {
    llvm::Value *LBVal = nullptr;
    llvm::Value *UBVal = nullptr;
    QualType IteratorTy = S.getIterationVariable()->getType();
    if (isOpenMPLoopBoundSharingDirective(S.getDirectiveKind())) {
      if (CGM.getLangOpts().OpenMPIsDevice && CGM.getTriple().isNVPTX()) {
        // normalize loop
        auto PrevLB = EmitLValue(S.getPrevLowerBoundVariable());
        llvm::Value *PrevLBVal = EmitLoadOfScalar(PrevLB, S.getLocStart());
        PrevLBVal = Builder.CreateIntCast(PrevLBVal, Builder.getIntNTy(IVSize),
                                          /*isSigned=*/false);
        LBVal = Builder.getIntN(IVSize, 0);
        UBVal = EmitLoadOfScalar(UB, /*Volatile=*/false, IteratorTy,
                                 S.getLocStart());
        UBVal = Builder.CreateSub(UBVal, PrevLBVal);
      } else {
        LBVal = EmitLoadOfScalar(LB, /*Volatile=*/false, IteratorTy,
                                 S.getLocStart());
        UBVal = EmitLoadOfScalar(UB, /*Volatile=*/false, IteratorTy,
                                 S.getLocStart());
      }
    } else {
      LBVal = Builder.getIntN(IVSize, 0);
      UBVal = EmitScalarExpr(S.getLastIteration());
    }
    RT.emitForDispatchInit(*this, S.getLocStart(), ScheduleKind, IVSize,
                           IVSigned, Ordered, LBVal, UBVal, Chunk);
  } else {
    CGOpenMPRuntime::StaticRTInput StaticInit(IVSize, IVSigned, Ordered, IL, LB,
                                              UB, ST, Chunk);
    RT.emitForStaticInit(*this, S.getLocStart(), S.getDirectiveKind(),
                         ScheduleKind, StaticInit);
  }

  EmitOMPOuterLoop(DynamicOrOrdered, IsMonotonic, /* IsDistribute =*/false, S,
                   LoopScope, Ordered, LB, UB, ST, IL, Chunk,
                   [](CodeGenFunction &, PrePostActionTy &) {});
}

void CodeGenFunction::EmitOMPDistributeOuterLoop(
    OpenMPDistScheduleClauseKind ScheduleKind, const OMPLoopDirective &S,
    OMPPrivateScope &LoopScope, Address LB, Address UB, Address ST, Address IL,
    llvm::Value *Chunk, const RegionCodeGenTy &CodeGenDistributeLoopContent) {

  auto &RT = CGM.getOpenMPRuntime();

  // Emit outer loop.
  // Same behavior as a OMPForOuterLoop, except that schedule cannot be
  // dynamic
  //

  const Expr *IVExpr = S.getIterationVariable();
  const unsigned IVSize = getContext().getTypeSize(IVExpr->getType());
  const bool IVSigned = IVExpr->getType()->hasSignedIntegerRepresentation();

  CGOpenMPRuntime::StaticRTInput StaticInit(
      IVSize, IVSigned, /* Ordered = */ false, IL, LB, UB, ST, Chunk);
  RT.emitDistributeStaticInit(*this, S.getLocStart(), ScheduleKind, StaticInit);

  EmitOMPOuterLoop(/* DynamicOrOrdered = */ false, /* IsMonotonic = */ false,
                   /* IsDistribute = */ true, S, LoopScope,
                   /* Ordered = */ false, LB, UB, ST, IL, Chunk,
                   CodeGenDistributeLoopContent);
}

static void emitCommonOMPTeamsDirective(CodeGenFunction &CGF,
                                        const OMPExecutableDirective &S,
                                        OpenMPDirectiveKind InnermostKind,
                                        const RegionCodeGenTy &CodeGen,
                                        unsigned CaptureLevel = 1,
                                        unsigned ImplicitParamStop = 0) {
  const auto *CS = cast<CapturedStmt>(S.getAssociatedStmt());
  if (S.hasClausesOfKind<OMPDependClause>() &&
      isOpenMPTargetExecutionDirective(S.getDirectiveKind()))
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
  auto OutlinedFn = CGF.CGM.getOpenMPRuntime().emitTeamsOutlinedFunction(
      S, *CS->getCapturedDecl()->param_begin(), InnermostKind, CodeGen,
      CaptureLevel, ImplicitParamStop);

  const OMPNumTeamsClause *NT = S.getSingleClause<OMPNumTeamsClause>();
  const OMPThreadLimitClause *TL = S.getSingleClause<OMPThreadLimitClause>();
  if (NT || TL) {
    Expr *NumTeams = (NT) ? NT->getNumTeams() : nullptr;
    Expr *ThreadLimit = (TL) ? TL->getThreadLimit() : nullptr;

    CGF.CGM.getOpenMPRuntime().emitNumTeamsClause(CGF, NumTeams, ThreadLimit,
                                                  S.getLocStart());
  }

  OMPTeamsScope Scope(CGF, S);
  llvm::SmallVector<llvm::Value *, 16> CapturedVars;
  CGF.GenerateOpenMPCapturedVars(*CS, CapturedVars, CaptureLevel);
  CGF.CGM.getOpenMPRuntime().emitTeamsCall(CGF, S, S.getLocStart(), OutlinedFn,
                                           CapturedVars);
}

void CodeGenFunction::EmitOMPDistributeParallelForSimdDirective(
    const OMPDistributeParallelForSimdDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    auto &&CGParallelFor = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
      auto &&CGInlinedWorksharingLoop = [&S](CodeGenFunction &CGF,
                                             PrePostActionTy &) {
        CGF.EmitOMPWorksharingLoop(S);
      };
      emitCommonOMPParallelDirective(CGF, S, OMPD_simd,
                                     CGInlinedWorksharingLoop);
    };
    CGF.EmitOMPDistributeLoop(S, CGParallelFor);
  };
  OMPLexicalScope Scope(*this, S, /*AsInlined=*/true);
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_distribute, CodeGen,
                                              false);
}

void CodeGenFunction::EmitOMPDistributeSimdDirective(
    const OMPDistributeSimdDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    auto &&CGSimd = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
      CGF.EmitOMPSimdLoop(S, false);
    };
    CGF.EmitOMPDistributeLoop(S, CGSimd);
  };
  OMPLexicalScope Scope(*this, S, /*AsInlined=*/true);
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_distribute, CodeGen,
                                              false);
}

void CodeGenFunction::EmitOMPTeamsDistributeDirective(
    const OMPTeamsDistributeDirective &S) {
  auto &&CGDistributeInlined = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    OMPPrivateScope PrivateScope(CGF);
    CGF.EmitOMPReductionClauseInit(S, PrivateScope);
    // TODO: Do we need this here?
    CGF.CGM.getOpenMPRuntime().emitInitDSBlock(CGF);
    (void)PrivateScope.Privatize();
    auto &&CGDistributeLoop = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
      CGF.EmitOMPDistributeLoop(S, [](CodeGenFunction &, PrePostActionTy &) {});
    };
    CGF.CGM.getOpenMPRuntime().emitInlinedDirective(CGF, OMPD_distribute,
                                                    CGDistributeLoop,
                                                    /*HasCancel=*/false);
    CGF.EmitOMPReductionClauseFinal(S, OMPD_teams);
  };
  emitCommonOMPTeamsDirective(*this, S, OMPD_teams_distribute,
                              CGDistributeInlined, 1, 2);
  emitPostUpdateForReductionClause(
      *this, S, [](CodeGenFunction &) -> llvm::Value * { return nullptr; });
}

void CodeGenFunction::EmitOMPTeamsDistributeSimdDirective(
    const OMPTeamsDistributeSimdDirective &S) {
  auto &&CGDistributeInlined = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    OMPPrivateScope PrivateScope(CGF);
    CGF.EmitOMPReductionClauseInit(S, PrivateScope);
    // TODO: Do we need this here?
    CGF.CGM.getOpenMPRuntime().emitInitDSBlock(CGF);
    (void)PrivateScope.Privatize();
    auto &&CGDistributeLoop = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
      auto &&CGSimd = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
        CGF.EmitOMPSimdLoop(S, false);
      };
      CGF.EmitOMPDistributeLoop(S, CGSimd);
    };
    CGF.CGM.getOpenMPRuntime().emitInlinedDirective(CGF, OMPD_distribute,
                                                    CGDistributeLoop,
                                                    /*HasCancel=*/false);
    CGF.EmitOMPReductionClauseFinal(S, OMPD_teams);
  };
  emitCommonOMPTeamsDirective(*this, S, OMPD_teams_distribute_simd,
                              CGDistributeInlined, 1, 2);
  emitPostUpdateForReductionClause(
      *this, S, [](CodeGenFunction &) -> llvm::Value * { return nullptr; });
}

void CodeGenFunction::EmitOMPTeamsDistributeParallelForSimdDirective(
    const OMPTeamsDistributeParallelForSimdDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    OMPPrivateScope PrivateScope(CGF);
    CGF.EmitOMPReductionClauseInit(S, PrivateScope);
    (void)PrivateScope.Privatize();
    auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
      auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
        auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
          CGF.EmitOMPWorksharingLoop(S);
        };
        emitCommonOMPParallelDirective(CGF, S, OMPD_for, CodeGen);
      };
      CGF.EmitOMPDistributeLoop(S, CodeGen);
    };
    CGF.CGM.getOpenMPRuntime().emitInlinedDirective(CGF, OMPD_distribute,
                                                    CodeGen,
                                                    /*HasCancel=*/false);
    CGF.EmitOMPReductionClauseFinal(S, OMPD_teams);
  };
  emitCommonOMPTeamsDirective(*this, S, OMPD_teams_distribute_parallel_for_simd,
                              CodeGen, /*CaptureLevel=*/1,
                              /*ImplicitParamStop=*/2);
  emitPostUpdateForReductionClause(
      *this, S, [](CodeGenFunction &) -> llvm::Value * { return nullptr; });
}

void CodeGenFunction::EmitOMPTeamsDistributeParallelForDirective(
    const OMPTeamsDistributeParallelForDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    OMPPrivateScope PrivateScope(CGF);
    CGF.EmitOMPReductionClauseInit(S, PrivateScope);
    (void)PrivateScope.Privatize();
    auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
      auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
        auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
          OMPCancelStackRAII CancelRegion(
              CGF, OMPD_teams_distribute_parallel_for, S.hasCancel());
          CGF.EmitOMPWorksharingLoop(S);
        };
        emitCommonOMPParallelDirective(CGF, S, OMPD_for, CodeGen);
      };
      CGF.EmitOMPDistributeLoop(S, CodeGen);
    };
    CGF.CGM.getOpenMPRuntime().emitInlinedDirective(CGF, OMPD_distribute,
                                                    CodeGen,
                                                    /*HasCancel=*/false);
    CGF.EmitOMPReductionClauseFinal(S, OMPD_teams);
  };
  emitCommonOMPTeamsDirective(*this, S, OMPD_teams_distribute_parallel_for,
                              CodeGen, /*CaptureLevel=*/1,
                              /*ImplicitParamStop=*/2);
  emitPostUpdateForReductionClause(
      *this, S, [](CodeGenFunction &) -> llvm::Value * { return nullptr; });
}

namespace {
  struct ScheduleKindModifiersTy {
    OpenMPScheduleClauseKind Kind;
    OpenMPScheduleClauseModifier M1;
    OpenMPScheduleClauseModifier M2;
    ScheduleKindModifiersTy(OpenMPScheduleClauseKind Kind,
                            OpenMPScheduleClauseModifier M1,
                            OpenMPScheduleClauseModifier M2)
        : Kind(Kind), M1(M1), M2(M2) {}
  };
} // namespace

bool CodeGenFunction::EmitOMPWorksharingLoop(const OMPLoopDirective &S) {
  // Emit the loop iteration variable.
  auto IVExpr = cast<DeclRefExpr>(S.getIterationVariable());
  auto IVDecl = cast<VarDecl>(IVExpr->getDecl());
  EmitVarDecl(*IVDecl);

  // Emit the conditional lastprivate iteration variable, if required.
  if (S.getConditionalLastprivateIterVariable()) {
    auto CLIVExpr =
        dyn_cast<DeclRefExpr>(S.getConditionalLastprivateIterVariable());
    auto CLIVDecl = cast<VarDecl>(CLIVExpr->getDecl());
    EmitVarDecl(*CLIVDecl);
  }

  // Emit the iterations count variable.
  // If it is not a variable, Sema decided to calculate iterations count on each
  // iteration (e.g., it is foldable into a constant).
  if (auto LIExpr = dyn_cast<DeclRefExpr>(S.getLastIteration())) {
    EmitVarDecl(*cast<VarDecl>(LIExpr->getDecl()));
    // Emit calculation of the iterations count.
    EmitIgnoredExpr(S.getCalcLastIteration());
  }

  auto &RT = CGM.getOpenMPRuntime();

  bool HasLastprivateClause;
  // Check pre-condition.
  {
    OMPLoopScope PreInitScope(*this, S, /*EmitPreInit=*/true);
    // Skip the entire loop if we don't meet the precondition.
    // If the condition constant folds and can be elided, avoid emitting the
    // whole loop.
    bool CondConstant;
    llvm::BasicBlock *ContBlock = nullptr;
    if (ConstantFoldsToSimpleInteger(S.getPreCond(), CondConstant)) {
      if (!CondConstant)
        return false;
    } else {
      auto *ThenBlock = createBasicBlock("omp.precond.then");
      ContBlock = createBasicBlock("omp.precond.end");
      emitPreCond(*this, S, S.getPreCond(), ThenBlock, ContBlock,
                  getProfileCount(&S));
      EmitBlock(ThenBlock);
      incrementProfileCounter(&S);
    }

    bool Ordered = false;
    if (auto *OrderedClause = S.getSingleClause<OMPOrderedClause>()) {
      if (OrderedClause->getNumForLoops())
        RT.emitDoacrossInit(*this, S);
      else
        Ordered = true;
    }

    llvm::DenseSet<const Expr *> EmittedFinals;
    emitAlignedClause(*this, S);
    bool HasLinears = EmitOMPLinearClauseInit(S);
    // Emit helper vars inits.
    LValue LB = EmitOMPHelperVar(cast<DeclRefExpr>(S.getLowerBoundVariable()));
    LValue UB = EmitOMPHelperVar(cast<DeclRefExpr>(S.getUpperBoundVariable()));
    LValue ST = EmitOMPHelperVar(cast<DeclRefExpr>(S.getStrideVariable()));
    LValue IL = EmitOMPHelperVar(cast<DeclRefExpr>(S.getIsLastIterVariable()));

    if (isOpenMPLoopBoundSharingDirective(S.getDirectiveKind())) {
      // When composing distribute with for we need to use the pragma distribute
      // chunk lower and upper bounds rather than the whole loop iteration
      // space. Therefore we copy the bounds of the previous schedule into the
      // the current ones.
      const unsigned IVSize = getContext().getTypeSize(IVExpr->getType());
      LValue PrevLB = EmitLValue(S.getPrevLowerBoundVariable());
      LValue PrevUB = EmitLValue(S.getPrevUpperBoundVariable());
      auto PrevLBVal = EmitLoadOfScalar(PrevLB, S.getLocStart());
      PrevLBVal = Builder.CreateIntCast(PrevLBVal, Builder.getIntNTy(IVSize),
                                        /*isSigned=*/false);
      auto PrevUBVal = EmitLoadOfScalar(PrevUB, S.getLocStart());
      PrevUBVal = Builder.CreateIntCast(PrevUBVal, Builder.getIntNTy(IVSize),
                                        /*isSigned=*/false);
      EmitStoreOfScalar(PrevLBVal, LB);
      EmitStoreOfScalar(PrevUBVal, UB);
    }

    // Emit 'then' code.
    {
      OMPPrivateScope LoopScope(*this);
      if (EmitOMPFirstprivateClause(S, LoopScope) || HasLinears) {
        // Emit implicit barrier to synchronize threads and avoid data races on
        // initialization of firstprivate variables and post-update of
        // lastprivate variables.
        CGM.getOpenMPRuntime().emitBarrierCall(
            *this, S.getLocStart(), OMPD_unknown, /*EmitChecks=*/false,
            /*ForceSimpleCall=*/true);
      }
      EmitOMPPrivateClause(S, LoopScope);
      HasLastprivateClause = EmitOMPLastprivateClauseInit(S, LoopScope);
      EmitOMPReductionClauseInit(S, LoopScope);
      EmitOMPPrivateLoopCounters(S, LoopScope);
      EmitOMPLinearClause(S, LoopScope);
      (void)LoopScope.Privatize();

      // Detect the loop schedule kind and chunk.
      llvm::Value *Chunk = nullptr;
      bool ChunkSizeOne = false;
      OpenMPScheduleTy ScheduleKind;
      if (auto *C = S.getSingleClause<OMPScheduleClause>()) {
        ScheduleKind.Schedule = C->getScheduleKind();
        ScheduleKind.M1 = C->getFirstScheduleModifier();
        ScheduleKind.M2 = C->getSecondScheduleModifier();
        if (const auto *Ch = C->getChunkSize()) {
          llvm::APSInt Result;
          if (ConstantFoldsToSimpleInteger(Ch, Result)) {
            ChunkSizeOne = Result == 1;
          }
          Chunk = EmitScalarExpr(Ch);
          Chunk = EmitScalarConversion(Chunk, Ch->getType(),
                                       S.getIterationVariable()->getType(),
                                       S.getLocStart());
        }
      }
      const unsigned IVSize = getContext().getTypeSize(IVExpr->getType());
      const bool IVSigned = IVExpr->getType()->hasSignedIntegerRepresentation();
      if (RT.generateCoalescedSchedule(ScheduleKind.Schedule, ChunkSizeOne,
                                       Ordered)) {
        // For NVPTX and other GPU targets high performance is often achieved
        // if adjacent threads access memory in a coalesced manner.  This is
        // true for loops that access memory with stride one if a static
        // schedule with chunk size of 1 is used.  We generate such code
        // whenever the OpenMP standard gives us freedom to do so.
        //
        // This case is called if there is no schedule clause, with a
        // schedule(auto), or with a schedule(static,1).
        //
        // Codegen is optimized for this case.  Since chunk size is 1 we do not
        // need to generate the inner loop, i.e., the chunk iterator can be
        // removed.
        // while(UB = min(UB, GlobalUB), idx = LB, idx < UB) {
        //   BODY;
        //   LB = LB + ST;
        //   UB = UB + ST;
        // }
        if (!Chunk) // Force use of chunk=1
          Chunk = Builder.getIntN(IVSize, 1);
        if (isOpenMPSimdDirective(S.getDirectiveKind()))
          EmitOMPSimdInit(S, /*IsMonotonic=*/true);

        OpenMPScheduleTy LoopSchedule;
        LoopSchedule.Schedule = OMPC_SCHEDULE_static;
        CGOpenMPRuntime::StaticRTInput StaticInit(
            IVSize, IVSigned, Ordered, IL.getAddress(), LB.getAddress(),
            UB.getAddress(), ST.getAddress(), Chunk);
        RT.emitForStaticInit(*this, S.getLocStart(), S.getDirectiveKind(),
                             LoopSchedule, StaticInit);
        auto LoopExit =
            getJumpDestInCurrentScope(createBasicBlock("omp.loop.exit"));

        // IV = LB;
        EmitIgnoredExpr(S.getInit());
        EmitOMPInnerLoop(
            S, LoopScope.requiresCleanups(), S.getCond() /* IV < GlobalUB */,
            S.getInc() /* Unused */, S.getConditionalLastprivateIterInit(),
            [&S, LoopExit](CodeGenFunction &CGF) {
              CGF.EmitOMPLoopBody(S, LoopExit);
              CGF.EmitStopPoint(&S);
            },
            [&S](CodeGenFunction &CGF) {
              // LB = LB + Stride
              CGF.EmitIgnoredExpr(S.getNextLowerBound());
              // IV = LB;
              CGF.EmitIgnoredExpr(S.getInit());
            });
        EmitBlock(LoopExit.getBlock());
        // Tell the runtime we are done.
        RT.emitForStaticFinish(*this, S.getLocStart(), S.getDirectiveKind());
      } else if (RT.isStaticNonchunked(ScheduleKind.Schedule,
                                       /* Chunked */ Chunk != nullptr) &&
                 !Ordered) {
        // OpenMP 4.5, 2.7.1 Loop Construct, Description.
        // If the static schedule kind is specified or if the ordered clause is
        // specified, and if no monotonic modifier is specified, the effect will
        // be as if the monotonic modifier was specified.
        if (isOpenMPSimdDirective(S.getDirectiveKind()))
          EmitOMPSimdInit(S, /*IsMonotonic=*/true);
        // OpenMP [2.7.1, Loop Construct, Description, table 2-1]
        // When no chunk_size is specified, the iteration space is divided into
        // chunks that are approximately equal in size, and at most one chunk is
        // distributed to each thread. Note that the size of the chunks is
        // unspecified in this case.
        CGOpenMPRuntime::StaticRTInput StaticInit(
            IVSize, IVSigned, Ordered, IL.getAddress(), LB.getAddress(),
            UB.getAddress(), ST.getAddress());
        RT.emitForStaticInit(*this, S.getLocStart(), S.getDirectiveKind(),
                             ScheduleKind, StaticInit);
        auto LoopExit =
            getJumpDestInCurrentScope(createBasicBlock("omp.loop.exit"));
        // UB = min(UB, GlobalUB);
        EmitIgnoredExpr(S.getEnsureUpperBound());
        // IV = LB;
        EmitIgnoredExpr(S.getInit());
        // while (idx <= UB) { BODY; ++idx; }
        EmitOMPInnerLoop(S, LoopScope.requiresCleanups(), S.getCond(),
                         S.getInc(), S.getConditionalLastprivateIterInit(),
                         [&S, LoopExit](CodeGenFunction &CGF) {
                           CGF.EmitOMPLoopBody(S, LoopExit);
                           CGF.EmitStopPoint(&S);
                         },
                         [](CodeGenFunction &) {});
        EmitBlock(LoopExit.getBlock());
        // Tell the runtime we are done.
        auto &&CodeGen = [&S](CodeGenFunction &CGF) {
          CGF.CGM.getOpenMPRuntime().emitForStaticFinish(CGF, S.getLocEnd(),
                                                         S.getDirectiveKind());
        };
        OMPCancelStack.emitExit(*this, S.getDirectiveKind(), CodeGen);
      } else {
        const bool IsMonotonic =
            Ordered || ScheduleKind.Schedule == OMPC_SCHEDULE_static ||
            ScheduleKind.Schedule == OMPC_SCHEDULE_unknown ||
            ScheduleKind.M1 == OMPC_SCHEDULE_MODIFIER_monotonic ||
            ScheduleKind.M2 == OMPC_SCHEDULE_MODIFIER_monotonic;
        // Emit the outer loop, which requests its work chunk [LB..UB] from
        // runtime and runs the inner loop to process it.
        EmitOMPForOuterLoop(ScheduleKind, IsMonotonic, S, LoopScope, Ordered,
                            LB.getAddress(), UB.getAddress(), ST.getAddress(),
                            IL.getAddress(), Chunk);
      }
      if (isOpenMPSimdDirective(S.getDirectiveKind())) {
        EmitOMPSimdFinal(S,
                         [&](CodeGenFunction &CGF) -> llvm::Value * {
                           return CGF.Builder.CreateIsNotNull(
                               CGF.EmitLoadOfScalar(IL, S.getLocStart()));
                         });
      }
      EmitOMPReductionClauseFinal(S, isOpenMPSimdDirective(S.getDirectiveKind())
                                         ? OMPD_parallel_for_simd
                                         : OMPD_parallel);
      // Emit post-update of the reduction variables if IsLastIter != 0.
      emitPostUpdateForReductionClause(
          *this, S, [&](CodeGenFunction &CGF) -> llvm::Value * {
            return CGF.Builder.CreateIsNotNull(
                CGF.EmitLoadOfScalar(IL, S.getLocStart()));
          });
      // Emit final copy of the lastprivate variables if IsLastIter != 0.
      if (HasLastprivateClause)
        EmitOMPLastprivateClauseFinal(
            S, isOpenMPSimdDirective(S.getDirectiveKind()),
            Builder.CreateIsNotNull(EmitLoadOfScalar(IL, S.getLocStart())));
    }
    EmitOMPLinearClauseFinal(S, [&](CodeGenFunction &CGF) -> llvm::Value * {
      return CGF.Builder.CreateIsNotNull(
          CGF.EmitLoadOfScalar(IL, S.getLocStart()));
    });
    // We're now done with the loop, so jump to the continuation block.
    if (ContBlock) {
      EmitBranch(ContBlock);
      EmitBlock(ContBlock, true);
    }
  }
  return HasLastprivateClause;
}

void CodeGenFunction::EmitOMPForDirective(const OMPForDirective &S) {
  bool HasLastprivates = false;
  auto &&CodeGen = [&S, &HasLastprivates](CodeGenFunction &CGF,
                                          PrePostActionTy &) {
    OMPCancelStackRAII CancelRegion(CGF, OMPD_for, S.hasCancel());
    HasLastprivates = CGF.EmitOMPWorksharingLoop(S);
  };
  {
    OMPLexicalScope Scope(*this, S, /*AsInlined=*/true);
    CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_for, CodeGen,
                                                S.hasCancel());
  }

  // Emit an implicit barrier at the end.
  if (CGM.getOpenMPRuntime().requiresBarrier(S) ||
      !S.getSingleClause<OMPNowaitClause>() || HasLastprivates) {
    CGM.getOpenMPRuntime().emitBarrierCall(*this, S.getLocStart(), OMPD_for);
  }
}

void CodeGenFunction::EmitOMPForSimdDirective(const OMPForSimdDirective &S) {
  bool HasLastprivates = false;
  auto &&CodeGen = [&S, &HasLastprivates](CodeGenFunction &CGF,
                                          PrePostActionTy &) {
    HasLastprivates = CGF.EmitOMPWorksharingLoop(S);
  };
  {
    OMPLexicalScope Scope(*this, S, /*AsInlined=*/true);
    CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_simd, CodeGen);
  }

  // Emit an implicit barrier at the end.
  if (!S.getSingleClause<OMPNowaitClause>() || HasLastprivates) {
    CGM.getOpenMPRuntime().emitBarrierCall(*this, S.getLocStart(), OMPD_for);
  }
}

static LValue createSectionLVal(CodeGenFunction &CGF, QualType Ty,
                                const Twine &Name,
                                llvm::Value *Init = nullptr) {
  auto LVal = CGF.MakeAddrLValue(CGF.CreateMemTemp(Ty, Name), Ty);
  if (Init)
    CGF.EmitStoreThroughLValue(RValue::get(Init), LVal, /*isInit*/ true);
  return LVal;
}

void CodeGenFunction::EmitSections(const OMPExecutableDirective &S) {
  const auto *PCS = cast<CapturedStmt>(S.getAssociatedStmt());
  if (S.hasClausesOfKind<OMPDependClause>() &&
      isOpenMPTargetExecutionDirective(S.getDirectiveKind()))
    PCS = cast<CapturedStmt>(PCS->getCapturedStmt());
  auto *Stmt = PCS->getCapturedStmt();
  auto *CS = dyn_cast<CompoundStmt>(Stmt);
  bool HasLastprivates = false;
  auto &&CodeGen = [&S, Stmt, CS, &HasLastprivates](CodeGenFunction &CGF,
                                                    PrePostActionTy &) {
    auto &C = CGF.CGM.getContext();
    auto KmpInt32Ty = C.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/1);
    // Emit helper vars inits.
    LValue LB = createSectionLVal(CGF, KmpInt32Ty, ".omp.sections.lb.",
                                  CGF.Builder.getInt32(0));
    auto *GlobalUBVal = CS != nullptr ? CGF.Builder.getInt32(CS->size() - 1)
                                      : CGF.Builder.getInt32(0);
    LValue UB =
        createSectionLVal(CGF, KmpInt32Ty, ".omp.sections.ub.", GlobalUBVal);
    LValue ST = createSectionLVal(CGF, KmpInt32Ty, ".omp.sections.st.",
                                  CGF.Builder.getInt32(1));
    LValue IL = createSectionLVal(CGF, KmpInt32Ty, ".omp.sections.il.",
                                  CGF.Builder.getInt32(0));
    // Loop counter.
    LValue IV = createSectionLVal(CGF, KmpInt32Ty, ".omp.sections.iv.");
    // Emit the conditional lastprivate iteration variable, if required.
    Expr *CLIVExpr = nullptr;
    if (auto *OSD = dyn_cast<OMPSectionsDirective>(&S))
      CLIVExpr = OSD->getConditionalLastprivateIterVariable();
    else if (auto *OSD = dyn_cast<OMPParallelSectionsDirective>(&S))
      CLIVExpr = OSD->getConditionalLastprivateIterVariable();
    assert(CLIVExpr && "Expecting conditional lastprivate iteration variable");
    auto CLIVDRE = dyn_cast<DeclRefExpr>(CLIVExpr);
    assert(CLIVDRE && "Expecting conditional lastprivate iteration variable");
    auto CLIVDecl = cast<VarDecl>(CLIVDRE->getDecl());
    CGF.EmitVarDecl(*CLIVDecl);
    auto CLIVLVal = CGF.EmitLValue(CLIVDRE);
    OpaqueValueExpr IVRefExpr(S.getLocStart(), KmpInt32Ty, VK_LValue);
    CodeGenFunction::OpaqueValueMapping OpaqueIV(CGF, &IVRefExpr, IV);
    OpaqueValueExpr UBRefExpr(S.getLocStart(), KmpInt32Ty, VK_LValue);
    CodeGenFunction::OpaqueValueMapping OpaqueUB(CGF, &UBRefExpr, UB);
    OpaqueValueExpr CLIVRefExpr(S.getLocStart(), C.getSizeType(), VK_LValue);
    CodeGenFunction::OpaqueValueMapping OpaqueCLIV(CGF, &CLIVRefExpr, CLIVLVal);
    // Generate condition for loop.
    BinaryOperator Cond(&IVRefExpr, &UBRefExpr, BO_LE, C.BoolTy, VK_RValue,
                        OK_Ordinary, S.getLocStart(),
                        /*fpContractable=*/false);
    // Increment for loop counter.
    UnaryOperator Inc(&IVRefExpr, UO_PreInc, KmpInt32Ty, VK_RValue, OK_Ordinary,
                      S.getLocStart());
    // Generate conditional lastprivate iteration variable assignment.
    ImplicitCastExpr IVCastExpr(ImplicitCastExpr::OnStack, C.getSizeType(),
                                CK_IntegralCast, &IVRefExpr, VK_RValue);
    BinaryOperator CLIVAssign(&CLIVRefExpr, &IVCastExpr, BO_Assign,
                              C.getSizeType(), VK_RValue, OK_Ordinary,
                              S.getLocStart(),
                              /*fpContractable=*/false);
    auto BodyGen = [Stmt, CS, &S, &IV](CodeGenFunction &CGF) {
      // Iterate through all sections and emit a switch construct:
      // switch (IV) {
      //   case 0:
      //     <SectionStmt[0]>;
      //     break;
      // ...
      //   case <NumSection> - 1:
      //     <SectionStmt[<NumSection> - 1]>;
      //     break;
      // }
      // .omp.sections.exit:
      auto *ExitBB = CGF.createBasicBlock(".omp.sections.exit");
      auto *SwitchStmt =
          CGF.Builder.CreateSwitch(CGF.EmitLoadOfScalar(IV, S.getLocStart()),
                                   ExitBB, CS == nullptr ? 1 : CS->size());
      if (CS) {
        unsigned CaseNumber = 0;
        for (auto *SubStmt : CS->children()) {
          auto CaseBB = CGF.createBasicBlock(".omp.sections.case");
          CGF.EmitBlock(CaseBB);
          SwitchStmt->addCase(CGF.Builder.getInt32(CaseNumber), CaseBB);
          CGF.EmitStmt(SubStmt);
          CGF.EmitBranch(ExitBB);
          ++CaseNumber;
        }
      } else {
        auto CaseBB = CGF.createBasicBlock(".omp.sections.case");
        CGF.EmitBlock(CaseBB);
        SwitchStmt->addCase(CGF.Builder.getInt32(0), CaseBB);
        CGF.EmitStmt(Stmt);
        CGF.EmitBranch(ExitBB);
      }
      CGF.EmitBlock(ExitBB, /*IsFinished=*/true);
    };

    CodeGenFunction::OMPPrivateScope LoopScope(CGF);
    if (CGF.EmitOMPFirstprivateClause(S, LoopScope)) {
      // Emit implicit barrier to synchronize threads and avoid data races on
      // initialization of firstprivate variables and post-update of lastprivate
      // variables.
      CGF.CGM.getOpenMPRuntime().emitBarrierCall(
          CGF, S.getLocStart(), OMPD_unknown, /*EmitChecks=*/false,
          /*ForceSimpleCall=*/true);
    }
    CGF.EmitOMPPrivateClause(S, LoopScope);
    HasLastprivates = CGF.EmitOMPLastprivateClauseInit(S, LoopScope);
    CGF.EmitOMPReductionClauseInit(S, LoopScope);
    (void)LoopScope.Privatize();

    // Emit static non-chunked loop.
    OpenMPScheduleTy ScheduleKind;
    ScheduleKind.Schedule = OMPC_SCHEDULE_static;
    CGOpenMPRuntime::StaticRTInput StaticInit(
        /*IVSize=*/32, /*IVSigned=*/true, /*Ordered=*/false, IL.getAddress(),
        LB.getAddress(), UB.getAddress(), ST.getAddress());
    CGF.CGM.getOpenMPRuntime().emitForStaticInit(
        CGF, S.getLocStart(), S.getDirectiveKind(), ScheduleKind, StaticInit);
    // UB = min(UB, GlobalUB);
    auto *UBVal = CGF.EmitLoadOfScalar(UB, S.getLocStart());
    auto *MinUBGlobalUB = CGF.Builder.CreateSelect(
        CGF.Builder.CreateICmpSLT(UBVal, GlobalUBVal), UBVal, GlobalUBVal);
    CGF.EmitStoreOfScalar(MinUBGlobalUB, UB);
    // IV = LB;
    CGF.EmitStoreOfScalar(CGF.EmitLoadOfScalar(LB, S.getLocStart()), IV);
    // while (idx <= UB) { BODY; ++idx; }
    CGF.EmitOMPInnerLoop(S, /*RequiresCleanup=*/false, &Cond, &Inc,
                         /*LastprivateIterInitExpr=*/&CLIVAssign, BodyGen,
                         [](CodeGenFunction &) {});
    // Tell the runtime we are done.
    auto &&CodeGen = [&S](CodeGenFunction &CGF) {
      CGF.CGM.getOpenMPRuntime().emitForStaticFinish(CGF, S.getLocEnd(),
                                                     S.getDirectiveKind());
    };
    CGF.OMPCancelStack.emitExit(CGF, S.getDirectiveKind(), CodeGen);
    CGF.EmitOMPReductionClauseFinal(S, OMPD_parallel);
    // Emit post-update of the reduction variables if IsLastIter != 0.
    emitPostUpdateForReductionClause(
        CGF, S, [&](CodeGenFunction &CGF) -> llvm::Value * {
          return CGF.Builder.CreateIsNotNull(
              CGF.EmitLoadOfScalar(IL, S.getLocStart()));
        });

    // Emit final copy of the lastprivate variables if IsLastIter != 0.
    if (HasLastprivates)
      CGF.EmitOMPLastprivateClauseFinal(
          S, /*NoFinals=*/false,
          CGF.Builder.CreateIsNotNull(
              CGF.EmitLoadOfScalar(IL, S.getLocStart())));
  };

  bool HasCancel = false;
  if (auto *OSD = dyn_cast<OMPSectionsDirective>(&S))
    HasCancel = OSD->hasCancel();
  else if (auto *OPSD = dyn_cast<OMPParallelSectionsDirective>(&S))
    HasCancel = OPSD->hasCancel();
  OMPCancelStackRAII CancelRegion(*this, S.getDirectiveKind(), HasCancel);
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_sections, CodeGen,
                                              HasCancel);
  // Emit barrier for lastprivates only if 'sections' directive has 'nowait'
  // clause. Otherwise the barrier will be generated by the codegen for the
  // directive.
  if (HasLastprivates && S.getSingleClause<OMPNowaitClause>()) {
    // Emit implicit barrier to synchronize threads and avoid data races on
    // initialization of firstprivate variables.
    CGM.getOpenMPRuntime().emitBarrierCall(*this, S.getLocStart(),
                                           OMPD_unknown);
  }
}

void CodeGenFunction::EmitOMPSectionsDirective(const OMPSectionsDirective &S) {
  {
    OMPLexicalScope Scope(*this, S, /*AsInlined=*/true);
    EmitSections(S);
  }
  // Emit an implicit barrier at the end.
  if (!S.getSingleClause<OMPNowaitClause>()) {
    CGM.getOpenMPRuntime().emitBarrierCall(*this, S.getLocStart(),
                                           OMPD_sections);
  }
}

void CodeGenFunction::EmitOMPSectionDirective(const OMPSectionDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    CGF.EmitStmt(cast<CapturedStmt>(S.getAssociatedStmt())->getCapturedStmt());
  };
  OMPLexicalScope Scope(*this, S, /*AsInlined=*/true);
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_section, CodeGen,
                                              S.hasCancel());
}

void CodeGenFunction::EmitOMPSingleDirective(const OMPSingleDirective &S) {
  llvm::SmallVector<const Expr *, 8> CopyprivateVars;
  llvm::SmallVector<const Expr *, 8> DestExprs;
  llvm::SmallVector<const Expr *, 8> SrcExprs;
  llvm::SmallVector<const Expr *, 8> AssignmentOps;
  // Check if there are any 'copyprivate' clauses associated with this
  // 'single' construct.
  // Build a list of copyprivate variables along with helper expressions
  // (<source>, <destination>, <destination>=<source> expressions)
  for (const auto *C : S.getClausesOfKind<OMPCopyprivateClause>()) {
    CopyprivateVars.append(C->varlists().begin(), C->varlists().end());
    DestExprs.append(C->destination_exprs().begin(),
                     C->destination_exprs().end());
    SrcExprs.append(C->source_exprs().begin(), C->source_exprs().end());
    AssignmentOps.append(C->assignment_ops().begin(),
                         C->assignment_ops().end());
  }
  // Emit code for 'single' region along with 'copyprivate' clauses
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    Action.Enter(CGF);
    OMPPrivateScope SingleScope(CGF);
    (void)CGF.EmitOMPFirstprivateClause(S, SingleScope);
    CGF.EmitOMPPrivateClause(S, SingleScope);
    (void)SingleScope.Privatize();
    CGF.EmitStmt(cast<CapturedStmt>(S.getAssociatedStmt())->getCapturedStmt());
  };
  {
    OMPLexicalScope Scope(*this, S, /*AsInlined=*/true);
    CGM.getOpenMPRuntime().emitSingleRegion(*this, CodeGen, S.getLocStart(),
                                            CopyprivateVars, DestExprs,
                                            SrcExprs, AssignmentOps);
  }
  // Emit an implicit barrier at the end (to avoid data race on firstprivate
  // init or if no 'nowait' clause was specified and no 'copyprivate' clause).
  if (!S.getSingleClause<OMPNowaitClause>() && CopyprivateVars.empty()) {
    CGM.getOpenMPRuntime().emitBarrierCall(
        *this, S.getLocStart(),
        S.getSingleClause<OMPNowaitClause>() ? OMPD_unknown : OMPD_single);
  }
}

void CodeGenFunction::EmitOMPMasterDirective(const OMPMasterDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    Action.Enter(CGF);
    CGF.EmitStmt(cast<CapturedStmt>(S.getAssociatedStmt())->getCapturedStmt());
  };
  OMPLexicalScope Scope(*this, S, /*AsInlined=*/true);
  CGM.getOpenMPRuntime().emitMasterRegion(*this, CodeGen, S.getLocStart());
}

void CodeGenFunction::EmitOMPCriticalDirective(const OMPCriticalDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    Action.Enter(CGF);
    CGF.EmitStmt(cast<CapturedStmt>(S.getAssociatedStmt())->getCapturedStmt());
  };
  Expr *Hint = nullptr;
  if (auto *HintClause = S.getSingleClause<OMPHintClause>())
    Hint = HintClause->getHint();
  OMPLexicalScope Scope(*this, S, /*AsInlined=*/true);
  CGM.getOpenMPRuntime().emitCriticalRegion(*this,
                                            S.getDirectiveName().getAsString(),
                                            CodeGen, S.getLocStart(), Hint);
}

void CodeGenFunction::EmitOMPParallelForDirective(
    const OMPParallelForDirective &S) {
  // Emit directive as a combined directive that consists of two implicit
  // directives: 'parallel' with 'for' directive.
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    OMPCancelStackRAII CancelRegion(CGF, OMPD_parallel_for, S.hasCancel());
    CGF.EmitOMPWorksharingLoop(S);
  };
  emitCommonOMPParallelDirective(*this, S, OMPD_for, CodeGen);
}

void CodeGenFunction::EmitOMPParallelForSimdDirective(
    const OMPParallelForSimdDirective &S) {
  // Emit directive as a combined directive that consists of two implicit
  // directives: 'parallel' with 'for' directive.
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    CGF.EmitOMPWorksharingLoop(S);
  };
  emitCommonOMPParallelDirective(*this, S, OMPD_simd, CodeGen);
}

void CodeGenFunction::EmitOMPParallelSectionsDirective(
    const OMPParallelSectionsDirective &S) {
  // Emit directive as a combined directive that consists of two implicit
  // directives: 'parallel' with 'sections' directive.
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    OMPLexicalScope Scope(CGF, S);
    CGF.EmitSections(S);
  };
  emitCommonOMPParallelDirective(*this, S, OMPD_sections, CodeGen);
}

void CodeGenFunction::EmitOMPTaskBasedDirective(const OMPExecutableDirective &S,
                                                const RegionCodeGenTy &BodyGen,
                                                const TaskGenTy &TaskGen,
                                                OMPTaskDataTy &Data) {
  // Emit outlined function for task construct.
  auto CS = cast<CapturedStmt>(S.getAssociatedStmt());
  auto *I = CS->getCapturedDecl()->param_begin();
  auto *PartId = std::next(I);
  auto *TaskT = std::next(I, 4);
  // Check if the task is final
  if (const auto *Clause = S.getSingleClause<OMPFinalClause>()) {
    // If the condition constant folds and can be elided, try to avoid emitting
    // the condition and the dead arm of the if/else.
    auto *Cond = Clause->getCondition();
    bool CondConstant;
    if (ConstantFoldsToSimpleInteger(Cond, CondConstant))
      Data.Final.setInt(CondConstant);
    else
      Data.Final.setPointer(EvaluateExprAsBool(Cond));
  } else {
    // By default the task is not final.
    Data.Final.setInt(/*IntVal=*/false);
  }
  // Check if the task has 'priority' clause.
  if (const auto *Clause = S.getSingleClause<OMPPriorityClause>()) {
    auto *Prio = Clause->getPriority();
    Data.Priority.setInt(/*IntVal=*/true);
    Data.Priority.setPointer(EmitScalarConversion(
        EmitScalarExpr(Prio), Prio->getType(),
        getContext().getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/1),
        Prio->getExprLoc()));
  }
  // The first function argument for tasks is a thread id, the second one is a
  // part id (0 for tied tasks, >=0 for untied task).
  llvm::DenseSet<const VarDecl *> EmittedAsPrivate;
  // Get list of private variables.
  if (!isOpenMPTargetExecutionDirective(S.getDirectiveKind())) {
    for (const auto *C : S.getClausesOfKind<OMPPrivateClause>()) {
      auto IRef = C->varlist_begin();
      for (auto *IInit : C->private_copies()) {
        auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(*IRef)->getDecl());
        if (EmittedAsPrivate.insert(OrigVD->getCanonicalDecl()).second) {
          Data.PrivateVars.push_back(*IRef);
          Data.PrivateCopies.push_back(IInit);
        }
        ++IRef;
      }
    }
  }
  EmittedAsPrivate.clear();
  // Get list of firstprivate variables.
  for (const auto *C : S.getClausesOfKind<OMPFirstprivateClause>()) {
    auto IRef = C->varlist_begin();
    auto IElemInitRef = C->inits().begin();
    for (auto *IInit : C->private_copies()) {
      auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(*IRef)->getDecl());
      if (EmittedAsPrivate.insert(OrigVD->getCanonicalDecl()).second) {
        Data.FirstprivateVars.push_back(*IRef);
        Data.FirstprivateCopies.push_back(IInit);
        Data.FirstprivateInits.push_back(*IElemInitRef);
      }
      ++IRef;
      ++IElemInitRef;
    }
  }
  llvm::DenseMap<const VarDecl *, const DeclRefExpr *> LastprivateDstsOrigs;
  if (!isOpenMPTargetExecutionDirective(S.getDirectiveKind())) {
    // Get list of lastprivate variables (for taskloops).
    for (const auto *C : S.getClausesOfKind<OMPLastprivateClause>()) {
      auto IRef = C->varlist_begin();
      auto ID = C->destination_exprs().begin();
      for (auto *IInit : C->private_copies()) {
        auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(*IRef)->getDecl());
        if (EmittedAsPrivate.insert(OrigVD->getCanonicalDecl()).second) {
          Data.LastprivateVars.push_back(*IRef);
          Data.LastprivateCopies.push_back(IInit);
        }
        LastprivateDstsOrigs.insert(
            {cast<VarDecl>(cast<DeclRefExpr>(*ID)->getDecl()),
             cast<DeclRefExpr>(*IRef)});
        ++IRef;
        ++ID;
      }
    }
    SmallVector<const Expr *, 4> LHSs;
    SmallVector<const Expr *, 4> RHSs;
    for (const auto *C : S.getClausesOfKind<OMPReductionClause>()) {
      auto IPriv = C->privates().begin();
      auto IRed = C->reduction_ops().begin();
      auto ILHS = C->lhs_exprs().begin();
      auto IRHS = C->rhs_exprs().begin();
      for (const auto *Ref : C->varlists()) {
        Data.ReductionVars.emplace_back(Ref);
        Data.ReductionCopies.emplace_back(*IPriv);
        Data.ReductionOps.emplace_back(*IRed);
        LHSs.emplace_back(*ILHS);
        RHSs.emplace_back(*IRHS);
        std::advance(IPriv, 1);
        std::advance(IRed, 1);
        std::advance(ILHS, 1);
        std::advance(IRHS, 1);
      }
    }
    Data.Reductions = CGM.getOpenMPRuntime().emitTaskReductionInit(
        *this, S.getLocStart(), LHSs, RHSs, Data);
  }
  // Build list of dependences.
  for (const auto *C : S.getClausesOfKind<OMPDependClause>())
    for (auto *IRef : C->varlists())
      Data.Dependences.push_back(std::make_pair(C->getDependencyKind(), IRef));
  auto &&CodeGen = [&Data, &S, CS, &BodyGen, &LastprivateDstsOrigs](
                       CodeGenFunction &CGF, PrePostActionTy &Action) {
    // Set proper addresses for generated private copies.
    OMPPrivateScope Scope(CGF);
    if (!Data.PrivateVars.empty() || !Data.FirstprivateVars.empty() ||
        !Data.LastprivateVars.empty() || Data.HasImplicitTargetArrays) {
      enum { PrivatesParam = 2, CopyFnParam = 3 };
      auto *CopyFn = CGF.Builder.CreateLoad(
          CGF.GetAddrOfLocalVar(CS->getCapturedDecl()->getParam(3)));
      auto *PrivatesPtr = CGF.Builder.CreateLoad(
          CGF.GetAddrOfLocalVar(CS->getCapturedDecl()->getParam(2)));
      // Map privates.
      llvm::SmallVector<std::pair<const VarDecl *, Address>, 16> PrivatePtrs;
      llvm::SmallVector<llvm::Value *, 16> CallArgs;
      CallArgs.push_back(PrivatesPtr);
      for (auto *E : Data.PrivateVars) {
        auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
        Address PrivatePtr = CGF.CreateMemTemp(
            CGF.getContext().getPointerType(E->getType()), ".priv.ptr.addr");
        PrivatePtrs.push_back(std::make_pair(VD, PrivatePtr));
        CallArgs.push_back(PrivatePtr.getPointer());
      }
      for (auto *E : Data.FirstprivateVars) {
        auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
        Address PrivatePtr =
            CGF.CreateMemTemp(CGF.getContext().getPointerType(E->getType()),
                              ".firstpriv.ptr.addr");
        PrivatePtrs.push_back(std::make_pair(VD, PrivatePtr));
        CallArgs.push_back(PrivatePtr.getPointer());
      }
      for (auto *E : Data.LastprivateVars) {
        auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
        Address PrivatePtr =
            CGF.CreateMemTemp(CGF.getContext().getPointerType(E->getType()),
                              ".lastpriv.ptr.addr");
        PrivatePtrs.push_back(std::make_pair(VD, PrivatePtr));
        CallArgs.push_back(PrivatePtr.getPointer());
      }
      if (Data.HasImplicitTargetArrays) {
        // the copy function requires references to set the pointers to the
        // firstprivate
        // variables of the mapping arrays. Replace the variable declarations
        // used in the
        // task region caller with pointers.
        auto &Context = CGF.getContext();
        const ImplicitParamDecl *BasePtrsParam =
            Data.FirstprivateSimpleArrayImplicit
                [OMPTaskDataTy::ImplicitMapArray::OMP_BASE_PTRS];
        auto *PointerToBasePtrsParam = ImplicitParamDecl::Create(
            Context, /*DC=*/nullptr, BasePtrsParam->getLocation(),
            /*Id=*/nullptr, Context.getPointerType(BasePtrsParam->getType()),
            ImplicitParamDecl::ImplicitParamKind::Other);
        Data.FirstprivateRefsSimpleArrayImplicit
            [OMPTaskDataTy::ImplicitMapArray::OMP_BASE_PTRS] =
            PointerToBasePtrsParam;
        CGF.EmitVarDecl(*PointerToBasePtrsParam);
        const ImplicitParamDecl *PtrsParam =
            Data.FirstprivateSimpleArrayImplicit
                [OMPTaskDataTy::ImplicitMapArray::OMP_PTRS];
        auto *PointerToPtrsParam = ImplicitParamDecl::Create(
            Context, /*DC=*/nullptr, PtrsParam->getLocation(), /*Id=*/nullptr,
            Context.getPointerType(PtrsParam->getType()),
            ImplicitParamDecl::ImplicitParamKind::Other);
        Data.FirstprivateRefsSimpleArrayImplicit
            [OMPTaskDataTy::ImplicitMapArray::OMP_PTRS] = PointerToPtrsParam;
        CGF.EmitVarDecl(*PointerToPtrsParam);
        const ImplicitParamDecl *SizesParam =
            Data.FirstprivateSimpleArrayImplicit
                [OMPTaskDataTy::ImplicitMapArray::OMP_SIZES];
        auto *PointerToSizesParam = ImplicitParamDecl::Create(
            Context, /*DC=*/nullptr, SizesParam->getLocation(), /*Id=*/nullptr,
            Context.getPointerType(SizesParam->getType()),
            ImplicitParamDecl::ImplicitParamKind::Other);
        Data.FirstprivateRefsSimpleArrayImplicit
            [OMPTaskDataTy::ImplicitMapArray::OMP_SIZES] = PointerToSizesParam;
        CGF.EmitVarDecl(*PointerToSizesParam);
        CallArgs.emplace_back(
            CGF.GetAddrOfLocalVar(PointerToBasePtrsParam).getPointer());
        CallArgs.emplace_back(
            CGF.GetAddrOfLocalVar(PointerToPtrsParam).getPointer());
        CallArgs.emplace_back(
            CGF.GetAddrOfLocalVar(PointerToSizesParam).getPointer());
      }
      CGF.CGM.getOpenMPRuntime().emitOutlinedFunctionCall(CGF, S.getLocStart(),
                                                          CopyFn, CallArgs);
      for (auto &&Pair : LastprivateDstsOrigs) {
        auto *OrigVD = cast<VarDecl>(Pair.second->getDecl());
        DeclRefExpr DRE(
            const_cast<VarDecl *>(OrigVD),
            /*RefersToEnclosingVariableOrCapture=*/CGF.CapturedStmtInfo->lookup(
                OrigVD) != nullptr,
            Pair.second->getType(), VK_LValue, Pair.second->getExprLoc());
        Scope.addPrivate(Pair.first, [&CGF, &DRE]() {
          return CGF.EmitLValue(&DRE).getAddress();
        });
      }
      for (auto &&Pair : PrivatePtrs) {
        Address Replacement(CGF.Builder.CreateLoad(Pair.second),
                            CGF.getContext().getDeclAlign(Pair.first));
        Scope.addPrivate(Pair.first, [Replacement]() { return Replacement; });
      }
    }
    if (Data.Reductions) {
      OMPLexicalScope LexScope(CGF, S, /*AsInlined=*/true);
      ReductionCodeGen RedCG(Data.ReductionVars, Data.ReductionCopies,
                             Data.ReductionOps);
      llvm::Value *ReductionsPtr = CGF.Builder.CreateLoad(
          CGF.GetAddrOfLocalVar(CS->getCapturedDecl()->getParam(9)));
      for (unsigned Cnt = 0, E = Data.ReductionVars.size(); Cnt < E; ++Cnt) {
        RedCG.emitSharedLValue(CGF, Cnt);
        RedCG.emitAggregateType(CGF, Cnt);
        Address Replacement = CGF.CGM.getOpenMPRuntime().getTaskReductionItem(
            CGF, S.getLocStart(), ReductionsPtr, RedCG.getSharedLValue(Cnt));
        Replacement =
            Address(CGF.EmitScalarConversion(
                        Replacement.getPointer(), CGF.getContext().VoidPtrTy,
                        CGF.getContext().getPointerType(
                            Data.ReductionCopies[Cnt]->getType()),
                        Data.ReductionVars[Cnt]->getExprLoc()),
                    Replacement.getAlignment());
        Replacement = RedCG.adjustPrivateAddress(CGF, Cnt, Replacement);
        Scope.addPrivate(RedCG.getBaseDecl(Cnt),
                         [Replacement]() { return Replacement; });
        // FIXME: This must removed once the runtime library is fixed.
        // Emit required threadprivate variables for
        // initilizer/combiner/finalizer.
        CGF.CGM.getOpenMPRuntime().emitTaskReductionFixups(CGF, S.getLocStart(),
                                                           RedCG, Cnt);
      }
    }
    // Privatize all private variables except for in_reduction items.
    (void)Scope.Privatize();
    SmallVector<const Expr *, 4> InRedVars;
    SmallVector<const Expr *, 4> InRedPrivs;
    SmallVector<const Expr *, 4> InRedOps;
    SmallVector<const Expr *, 4> TaskgroupDescriptors;
    for (const auto *C : S.getClausesOfKind<OMPInReductionClause>()) {
      auto IPriv = C->privates().begin();
      auto IRed = C->reduction_ops().begin();
      auto ITD = C->taskgroup_descriptors().begin();
      for (const auto *Ref : C->varlists()) {
        InRedVars.emplace_back(Ref);
        InRedPrivs.emplace_back(*IPriv);
        InRedOps.emplace_back(*IRed);
        TaskgroupDescriptors.emplace_back(*ITD);
        std::advance(IPriv, 1);
        std::advance(IRed, 1);
        std::advance(ITD, 1);
      }
    }
    // Privatize in_reduction items here, because taskgroup descriptors must be
    // privatized earlier.
    OMPPrivateScope InRedScope(CGF);
    if (!InRedVars.empty()) {
      ReductionCodeGen RedCG(InRedVars, InRedPrivs, InRedOps);
      for (unsigned Cnt = 0, E = InRedVars.size(); Cnt < E; ++Cnt) {
        RedCG.emitSharedLValue(CGF, Cnt);
        RedCG.emitAggregateType(CGF, Cnt);
        // The taskgroup descriptor variable is always implicit firstprivate and
        // privatized already during procoessing of the firstprivates.
        llvm::Value *ReductionsPtr =
            CGF.EmitLoadOfScalar(CGF.EmitLValue(TaskgroupDescriptors[Cnt]),
                                 InRedVars[Cnt]->getExprLoc());
        Address Replacement = CGF.CGM.getOpenMPRuntime().getTaskReductionItem(
            CGF, S.getLocStart(), ReductionsPtr, RedCG.getSharedLValue(Cnt));
        Replacement = Address(
            CGF.EmitScalarConversion(
                Replacement.getPointer(), CGF.getContext().VoidPtrTy,
                CGF.getContext().getPointerType(InRedPrivs[Cnt]->getType()),
                InRedVars[Cnt]->getExprLoc()),
            Replacement.getAlignment());
        Replacement = RedCG.adjustPrivateAddress(CGF, Cnt, Replacement);
        InRedScope.addPrivate(RedCG.getBaseDecl(Cnt),
                              [Replacement]() { return Replacement; });
        // FIXME: This must removed once the runtime library is fixed.
        // Emit required threadprivate variables for
        // initilizer/combiner/finalizer.
        CGF.CGM.getOpenMPRuntime().emitTaskReductionFixups(CGF, S.getLocStart(),
                                                           RedCG, Cnt);
      }
    }
    (void)InRedScope.Privatize();

    Action.Enter(CGF);
    OMPLexicalScope OMPScope(
        CGF, S, isOpenMPTargetExecutionDirective(S.getDirectiveKind()),
        /*EmitPreInit=*/false);
    BodyGen(CGF);
  };

  auto *OutlinedFn = CGM.getOpenMPRuntime().emitTaskOutlinedFunction(
      S, *I, *PartId, *TaskT, S.getDirectiveKind(), CodeGen, Data.Tied,
      Data.NumberOfParts);
  OMPLexicalScope Scope(
      *this, S, /*AsInlined=*/false,
      /*EmitPreInit=*/!isOpenMPTargetExecutionDirective(S.getDirectiveKind()));
  TaskGen(*this, OutlinedFn, Data);
}

void CodeGenFunction::EmitOMPTaskDirective(const OMPTaskDirective &S) {
  // Emit outlined function for task construct.
  auto CS = cast<CapturedStmt>(S.getAssociatedStmt());
  auto CapturedStruct = GenerateCapturedStmtArgument(*CS);
  auto SharedsTy = getContext().getRecordType(CS->getCapturedRecordDecl());
  const Expr *IfCond = nullptr;
  for (const auto *C : S.getClausesOfKind<OMPIfClause>()) {
    if (C->getNameModifier() == OMPD_unknown ||
        C->getNameModifier() == OMPD_task) {
      IfCond = C->getCondition();
      break;
    }
  }

  OMPTaskDataTy Data;
  // Check if we should emit tied or untied task.
  Data.Tied = !S.getSingleClause<OMPUntiedClause>();
  auto &&BodyGen = [CS](CodeGenFunction &CGF, PrePostActionTy &) {
    CGF.EmitStmt(CS->getCapturedStmt());
  };
  auto &&TaskGen = [&S, SharedsTy, CapturedStruct,
                    IfCond](CodeGenFunction &CGF, llvm::Value *OutlinedFn,
                            const OMPTaskDataTy &Data) {
    CGF.CGM.getOpenMPRuntime().emitTaskCall(CGF, S.getLocStart(), S, OutlinedFn,
                                            SharedsTy, CapturedStruct, IfCond,
                                            Data);
  };
  EmitOMPTaskBasedDirective(S, BodyGen, TaskGen, Data);
}

void CodeGenFunction::EmitOMPTaskyieldDirective(
    const OMPTaskyieldDirective &S) {
  CGM.getOpenMPRuntime().emitTaskyieldCall(*this, S.getLocStart());
}

void CodeGenFunction::EmitOMPBarrierDirective(const OMPBarrierDirective &S) {
  CGM.getOpenMPRuntime().emitBarrierCall(*this, S.getLocStart(), OMPD_barrier);
}

void CodeGenFunction::EmitOMPTaskwaitDirective(const OMPTaskwaitDirective &S) {
  CGM.getOpenMPRuntime().emitTaskwaitCall(*this, S.getLocStart());
}

void CodeGenFunction::EmitOMPTaskgroupDirective(
    const OMPTaskgroupDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    Action.Enter(CGF);
    if (const Expr *E = S.getReductionRef()) {
      SmallVector<const Expr *, 4> LHSs;
      SmallVector<const Expr *, 4> RHSs;
      OMPTaskDataTy Data;
      for (const auto *C : S.getClausesOfKind<OMPTaskReductionClause>()) {
        auto IPriv = C->privates().begin();
        auto IRed = C->reduction_ops().begin();
        auto ILHS = C->lhs_exprs().begin();
        auto IRHS = C->rhs_exprs().begin();
        for (const auto *Ref : C->varlists()) {
          Data.ReductionVars.emplace_back(Ref);
          Data.ReductionCopies.emplace_back(*IPriv);
          Data.ReductionOps.emplace_back(*IRed);
          LHSs.emplace_back(*ILHS);
          RHSs.emplace_back(*IRHS);
          std::advance(IPriv, 1);
          std::advance(IRed, 1);
          std::advance(ILHS, 1);
          std::advance(IRHS, 1);
        }
      }
      llvm::Value *ReductionDesc =
          CGF.CGM.getOpenMPRuntime().emitTaskReductionInit(CGF, S.getLocStart(),
                                                           LHSs, RHSs, Data);
      const auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
      CGF.EmitVarDecl(*VD);
      CGF.EmitStoreOfScalar(ReductionDesc, CGF.GetAddrOfLocalVar(VD),
                            /*Volatile=*/false, E->getType());
    }
    CGF.EmitStmt(cast<CapturedStmt>(S.getAssociatedStmt())->getCapturedStmt());
  };
  OMPLexicalScope Scope(*this, S, /*AsInlined=*/true);
  CGM.getOpenMPRuntime().emitTaskgroupRegion(*this, CodeGen, S.getLocStart());
}

void CodeGenFunction::EmitOMPFlushDirective(const OMPFlushDirective &S) {
  CGM.getOpenMPRuntime().emitFlush(*this, [&]() -> ArrayRef<const Expr *> {
    if (const auto *FlushClause = S.getSingleClause<OMPFlushClause>()) {
      return llvm::makeArrayRef(FlushClause->varlist_begin(),
                                FlushClause->varlist_end());
    }
    return llvm::None;
  }(), S.getLocStart());
}

void CodeGenFunction::EmitOMPLastprivateUpdateDirective(
    const OMPLastprivateUpdateDirective &S) {
  EmitStmt(cast<CapturedStmt>(S.getAssociatedStmt())->getCapturedStmt());
  if (const auto *LastprivateUpdateClause =
          S.getSingleClause<OMPLastprivateUpdateClause>()) {
    for (auto *Expr : LastprivateUpdateClause->varlists())
      EmitIgnoredExpr(Expr);
  }
}

void CodeGenFunction::EmitOMPDistributeLoop(
    const OMPLoopDirective &S,
    const RegionCodeGenTy &CodeGenDistributeLoopContent) {
  // Emit the loop iteration variable.
  auto IVExpr = cast<DeclRefExpr>(S.getIterationVariable());
  auto IVDecl = cast<VarDecl>(IVExpr->getDecl());
  EmitVarDecl(*IVDecl);

  // Emit the iterations count variable.
  // If it is not a variable, Sema decided to calculate iterations count on each
  // iteration (e.g., it is foldable into a constant).
  if (auto LIExpr = dyn_cast<DeclRefExpr>(S.getLastIteration())) {
    EmitVarDecl(*cast<VarDecl>(LIExpr->getDecl()));
    // Emit calculation of the iterations count.
    EmitIgnoredExpr(S.getCalcLastIteration());
  }

  auto &RT = CGM.getOpenMPRuntime();

  bool HasLastprivateClause = false;
  // Check pre-condition.
  {
    OMPLoopScope PreInitScope(*this, S, /*EmitPreInit=*/true);
    // Skip the entire loop if we don't meet the precondition.
    // If the condition constant folds and can be elided, avoid emitting the
    // whole loop.
    bool CondConstant;
    llvm::BasicBlock *ContBlock = nullptr;
    if (ConstantFoldsToSimpleInteger(S.getPreCond(), CondConstant)) {
      if (!CondConstant)
        return;
    } else {
      auto *ThenBlock = createBasicBlock("omp.precond.then");
      ContBlock = createBasicBlock("omp.precond.end");
      emitPreCond(*this, S, S.getPreCond(), ThenBlock, ContBlock,
                  getProfileCount(&S));
      EmitBlock(ThenBlock);
      incrementProfileCounter(&S);
    }

    // Emit 'then' code.
    {
      // Emit helper vars inits.
      LValue LB =
          EmitOMPHelperVar(cast<DeclRefExpr>(S.getLowerBoundVariable()));
      LValue UB =
          EmitOMPHelperVar(cast<DeclRefExpr>(S.getUpperBoundVariable()));
      LValue ST = EmitOMPHelperVar(cast<DeclRefExpr>(S.getStrideVariable()));
      LValue IL =
          EmitOMPHelperVar(cast<DeclRefExpr>(S.getIsLastIterVariable()));

      OMPPrivateScope LoopScope(*this);
      if (EmitOMPFirstprivateClause(S, LoopScope)) {
        // Emit implicit barrier to synchronize threads and avoid data races on
        // initialization of firstprivate variables and post-update of
        // lastprivate variables.
        CGM.getOpenMPRuntime().emitBarrierCall(
          *this, S.getLocStart(), OMPD_unknown, /*EmitChecks=*/false,
          /*ForceSimpleCall=*/true);
      }
      EmitOMPPrivateClause(S, LoopScope);
      HasLastprivateClause = EmitOMPLastprivateClauseInit(S, LoopScope);
      EmitOMPPrivateLoopCounters(S, LoopScope);
      (void)LoopScope.Privatize();

      // Detect the distribute schedule kind and chunk.
      llvm::Value *DistChunk = nullptr;
      OpenMPDistScheduleClauseKind DistScheduleKind =
          OMPC_DIST_SCHEDULE_unknown;
      if (auto *C = S.getSingleClause<OMPDistScheduleClause>()) {
        DistScheduleKind = C->getDistScheduleKind();
        if (const auto *Ch = C->getChunkSize()) {
          DistChunk = EmitScalarExpr(Ch);
          DistChunk = EmitScalarConversion(DistChunk, Ch->getType(),
                                           S.getIterationVariable()->getType(),
                                           S.getLocStart());
        }
      }
      const unsigned IVSize = getContext().getTypeSize(IVExpr->getType());
      const bool IVSigned = IVExpr->getType()->hasSignedIntegerRepresentation();

      // Detect the loop schedule kind and chunk.
      llvm::Value *Chunk = nullptr;
      bool ChunkSizeOne = false;
      OpenMPScheduleClauseKind ScheduleKind = OMPC_SCHEDULE_unknown;
      OpenMPScheduleClauseModifier M1 = OMPC_SCHEDULE_MODIFIER_unknown;
      OpenMPScheduleClauseModifier M2 = OMPC_SCHEDULE_MODIFIER_unknown;
      if (auto *C = S.getSingleClause<OMPScheduleClause>()) {
        ScheduleKind = C->getScheduleKind();
        M1 = C->getFirstScheduleModifier();
        M2 = C->getSecondScheduleModifier();
        if (const auto *Ch = C->getChunkSize()) {
          llvm::APSInt Result;
          if (ConstantFoldsToSimpleInteger(Ch, Result)) {
            ChunkSizeOne = Result == 1;
          }
          Chunk = EmitScalarExpr(Ch);
          Chunk = EmitScalarConversion(Chunk, Ch->getType(),
                                       S.getIterationVariable()->getType(),
                                       S.getLocStart());
        }
      }
      const bool Ordered = S.getSingleClause<OMPOrderedClause>() != nullptr;

      if (RT.generateCoalescedSchedule(DistScheduleKind, ScheduleKind,
                                       /* Chunked */ DistChunk != nullptr,
                                       ChunkSizeOne, Ordered)) {
        // For NVPTX and other GPU targets high performance is often achieved
        // if adjacent threads access memory in a coalesced manner.  This is
        // true for loops that access memory with stride one if a static
        // schedule with chunk size of 1 is used.  We generate such code
        // whenever the OpenMP standard gives us freedom to do so.
        //
        // This case is called if there is no dist_schedule clause, and there is
        // no schedule clause, with a schedule(auto), or with a
        // schedule(static,1).
        //
        // Codegen is optimized for this case.  Since chunk size is 1 we do not
        // need to generate the inner loop, i.e., the chunk iterator can be
        // removed.
        // while(UB = min(UB, GlobalUB), idx = LB, idx < UB) {
        //   BODY;
        //   LB = LB + ST;
        //   UB = UB + ST;
        // }
        if (!Chunk) // Force use of chunk = 1.
          Chunk = Builder.getIntN(IVSize, 1);
        CGOpenMPRuntime::StaticRTInput StaticInit(
            IVSize, IVSigned, /* Ordered = */ false, IL.getAddress(),
            LB.getAddress(), UB.getAddress(), ST.getAddress(), Chunk);
        RT.emitDistributeStaticInit(*this, S.getLocStart(), DistScheduleKind,
                                    StaticInit, /*CoalescedDistSchedule=*/true);
        auto LoopExit =
            getJumpDestInCurrentScope(createBasicBlock("omp.loop.exit"));
        // IV = LB
        EmitIgnoredExpr(S.getInit());
        EmitOMPInnerLoop(S, LoopScope.requiresCleanups(),
                         S.getDistCond() /* IV < GlobalUB */,
                         S.getInc() /* Unused */,
                         /*LastprivateIterInitExpr=*/nullptr,
                         [&S, &LoopExit](CodeGenFunction &CGF) {
                           CGF.EmitOMPLoopBody(S, LoopExit);
                           CGF.EmitStopPoint(&S);
                         },
                         [&S](CodeGenFunction &CGF) {
                           // LB = LB + Stride
                           CGF.EmitIgnoredExpr(S.getNextLowerBound());
                           // IV = LB;
                           CGF.EmitIgnoredExpr(S.getInit());
                         });
        EmitBlock(LoopExit.getBlock());
        // Tell the runtime we are done.
        RT.emitForStaticFinish(*this, S.getLocStart(), S.getDirectiveKind(),
                               /*CoalescedDistSchedule=*/true);
        // Emit the parallel reduction.
        EmitOMPReductionClauseFinal(S, OMPD_distribute_parallel_for);
      } else if (RT.isStaticNonchunked(DistScheduleKind,
                                       /* Chunked */ DistChunk != nullptr)) {
        // OpenMP [2.10.8, distribute Construct, Description]
        // If dist_schedule is specified, kind must be static. If specified,
        // iterations are divided into chunks of size chunk_size, chunks are
        // assigned to the teams of the league in a round-robin fashion in the
        // order of the team number. When no chunk_size is specified, the
        // iteration space is divided into chunks that are approximately equal
        // in size, and at most one chunk is distributed to each team of the
        // league. The size of the chunks is unspecified in this case.
        CGOpenMPRuntime::StaticRTInput StaticInit(
            IVSize, IVSigned, /* Ordered = */ false, IL.getAddress(),
            LB.getAddress(), UB.getAddress(), ST.getAddress());
        RT.emitDistributeStaticInit(*this, S.getLocStart(), DistScheduleKind,
                                    StaticInit);
        auto LoopExit =
            getJumpDestInCurrentScope(createBasicBlock("omp.loop.exit"));
        // UB = min(UB, GlobalUB);
        EmitIgnoredExpr(S.getEnsureUpperBound());
        // IV = LB;
        EmitIgnoredExpr(S.getInit());
        // while (idx <= UB) { BODY; ++idx; }
        EmitOMPInnerLoop(S, LoopScope.requiresCleanups(),
                         isOpenMPLoopBoundSharingDirective(S.getDirectiveKind())
                             ? S.getDistCond()
                             : S.getCond(),
                         isOpenMPLoopBoundSharingDirective(S.getDirectiveKind())
                             ? S.getDistInc()
                             : S.getInc(),
                         /*LastprivateIterInitExpr=*/nullptr,
                         [&S, &LoopExit,
                          &CodeGenDistributeLoopContent](CodeGenFunction &CGF) {
                           if (S.getDirectiveKind() == OMPD_distribute ||
                               S.getDirectiveKind() == OMPD_teams_distribute ||
                               S.getDirectiveKind() ==
                                   OMPD_target_teams_distribute) {
                             CGF.EmitOMPLoopBody(S, LoopExit);
                             CGF.EmitStopPoint(&S);
                           } else if (isOpenMPLoopBoundSharingDirective(
                                          S.getDirectiveKind())) {
                             CodeGenDistributeLoopContent(CGF);
                           }
                         },
                         [](CodeGenFunction &) {});
        EmitBlock(LoopExit.getBlock());
        // Tell the runtime we are done.
        RT.emitForStaticFinish(*this, S.getLocStart(), S.getDirectiveKind());
      } else {
        // Emit the outer loop, which requests its work chunk [LB..UB] from
        // runtime and runs the inner loop to process it.
        EmitOMPDistributeOuterLoop(DistScheduleKind, S, LoopScope,
                                   LB.getAddress(), UB.getAddress(),
                                   ST.getAddress(), IL.getAddress(), DistChunk,
                                   CodeGenDistributeLoopContent);
      }

      // Emit final copy of the lastprivate variables if IsLastIter != 0.
      if (HasLastprivateClause)
        EmitOMPLastprivateClauseFinal(
            S, isOpenMPSimdDirective(S.getDirectiveKind()),
            Builder.CreateIsNotNull(EmitLoadOfScalar(IL, S.getLocStart())));
    }

    // We're now done with the loop, so jump to the continuation block.
    if (ContBlock) {
      EmitBranch(ContBlock);
      EmitBlock(ContBlock, true);
    }
  }
}

void CodeGenFunction::EmitOMPDistributeDirective(
    const OMPDistributeDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    CGF.EmitOMPDistributeLoop(S, [](CodeGenFunction &, PrePostActionTy &) {});
  };
  OMPLexicalScope Scope(*this, S, /*AsInlined=*/true);
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_distribute, CodeGen,
                                              false);
}

void CodeGenFunction::EmitOMPDistributeParallelForDirective(
    const OMPDistributeParallelForDirective &S) {
  auto &&CGParallelFor = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    auto &&CGInlinedWorksharingLoop = [&S](CodeGenFunction &CGF,
                                           PrePostActionTy &) {
      OMPCancelStackRAII CancelRegion(CGF, S.getDirectiveKind(), S.hasCancel());
      CGF.EmitOMPWorksharingLoop(S);
    };
    emitCommonOMPParallelDirective(CGF, S, OMPD_for, CGInlinedWorksharingLoop);
  };

  auto &&CodeGen = [&S, &CGParallelFor](CodeGenFunction &CGF,
                                        PrePostActionTy &) {
    CGF.EmitOMPDistributeLoop(S, CGParallelFor);
  };

  OMPLexicalScope Scope(*this, S, /*AsInlined=*/true);
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_distribute, CodeGen,
                                              false);
}

static llvm::Function *emitOutlinedOrderedFunction(CodeGenModule &CGM,
                                                   const CapturedStmt *S) {
  CodeGenFunction CGF(CGM, /*suppressNewContext=*/true);
  CodeGenFunction::CGCapturedStmtInfo CapStmtInfo;
  CGF.CapturedStmtInfo = &CapStmtInfo;
  auto *Fn = CGF.GenerateOpenMPCapturedStmtFunction(*S);
  Fn->addFnAttr(llvm::Attribute::NoInline);
  return Fn;
}

void CodeGenFunction::EmitOMPOrderedDirective(const OMPOrderedDirective &S) {
  if (!S.getAssociatedStmt()) {
    for (const auto *DC : S.getClausesOfKind<OMPDependClause>())
      CGM.getOpenMPRuntime().emitDoacrossOrdered(*this, DC);
    return;
  }
  auto *C = S.getSingleClause<OMPSIMDClause>();
  auto &&CodeGen = [&S, C, this](CodeGenFunction &CGF,
                                 PrePostActionTy &Action) {
    if (C) {
      auto CS = cast<CapturedStmt>(S.getAssociatedStmt());
      llvm::SmallVector<llvm::Value *, 16> CapturedVars;
      CGF.GenerateOpenMPCapturedVars(*CS, CapturedVars);
      auto *OutlinedFn = emitOutlinedOrderedFunction(CGM, CS);
      CGM.getOpenMPRuntime().emitOutlinedFunctionCall(CGF, S.getLocStart(),
                                                      OutlinedFn, CapturedVars);
    } else {
      Action.Enter(CGF);
      CGF.EmitStmt(
          cast<CapturedStmt>(S.getAssociatedStmt())->getCapturedStmt());
    }
  };
  OMPLexicalScope Scope(*this, S, /*AsInlined=*/true);
  CGM.getOpenMPRuntime().emitOrderedRegion(*this, CodeGen, S.getLocStart(), !C);
}

static llvm::Value *convertToScalarValue(CodeGenFunction &CGF, RValue Val,
                                         QualType SrcType, QualType DestType,
                                         SourceLocation Loc) {
  assert(CGF.hasScalarEvaluationKind(DestType) &&
         "DestType must have scalar evaluation kind.");
  assert(!Val.isAggregate() && "Must be a scalar or complex.");
  return Val.isScalar()
             ? CGF.EmitScalarConversion(Val.getScalarVal(), SrcType, DestType,
                                        Loc)
             : CGF.EmitComplexToScalarConversion(Val.getComplexVal(), SrcType,
                                                 DestType, Loc);
}

static CodeGenFunction::ComplexPairTy
convertToComplexValue(CodeGenFunction &CGF, RValue Val, QualType SrcType,
                      QualType DestType, SourceLocation Loc) {
  assert(CGF.getEvaluationKind(DestType) == TEK_Complex &&
         "DestType must have complex evaluation kind.");
  CodeGenFunction::ComplexPairTy ComplexVal;
  if (Val.isScalar()) {
    // Convert the input element to the element type of the complex.
    auto DestElementType = DestType->castAs<ComplexType>()->getElementType();
    auto ScalarVal = CGF.EmitScalarConversion(Val.getScalarVal(), SrcType,
                                              DestElementType, Loc);
    ComplexVal = CodeGenFunction::ComplexPairTy(
        ScalarVal, llvm::Constant::getNullValue(ScalarVal->getType()));
  } else {
    assert(Val.isComplex() && "Must be a scalar or complex.");
    auto SrcElementType = SrcType->castAs<ComplexType>()->getElementType();
    auto DestElementType = DestType->castAs<ComplexType>()->getElementType();
    ComplexVal.first = CGF.EmitScalarConversion(
        Val.getComplexVal().first, SrcElementType, DestElementType, Loc);
    ComplexVal.second = CGF.EmitScalarConversion(
        Val.getComplexVal().second, SrcElementType, DestElementType, Loc);
  }
  return ComplexVal;
}

static void emitSimpleAtomicStore(CodeGenFunction &CGF, bool IsSeqCst,
                                  LValue LVal, RValue RVal) {
  if (LVal.isGlobalReg()) {
    CGF.EmitStoreThroughGlobalRegLValue(RVal, LVal);
  } else {
    CGF.EmitAtomicStore(RVal, LVal,
                        IsSeqCst ? llvm::AtomicOrdering::SequentiallyConsistent
                                 : llvm::AtomicOrdering::Monotonic,
                        LVal.isVolatile(), /*IsInit=*/false);
  }
}

void CodeGenFunction::emitOMPSimpleStore(LValue LVal, RValue RVal,
                                         QualType RValTy, SourceLocation Loc) {
  switch (getEvaluationKind(LVal.getType())) {
  case TEK_Scalar:
    EmitStoreThroughLValue(RValue::get(convertToScalarValue(
                               *this, RVal, RValTy, LVal.getType(), Loc)),
                           LVal);
    break;
  case TEK_Complex:
    EmitStoreOfComplex(
        convertToComplexValue(*this, RVal, RValTy, LVal.getType(), Loc), LVal,
        /*isInit=*/false);
    break;
  case TEK_Aggregate:
    llvm_unreachable("Must be a scalar or complex.");
  }
}

static void EmitOMPAtomicReadExpr(CodeGenFunction &CGF, bool IsSeqCst,
                                  const Expr *X, const Expr *V,
                                  SourceLocation Loc) {
  // v = x;
  assert(V->isLValue() && "V of 'omp atomic read' is not lvalue");
  assert(X->isLValue() && "X of 'omp atomic read' is not lvalue");
  LValue XLValue = CGF.EmitLValue(X);
  LValue VLValue = CGF.EmitLValue(V);
  RValue Res = XLValue.isGlobalReg()
                   ? CGF.EmitLoadOfLValue(XLValue, Loc)
                   : CGF.EmitAtomicLoad(
                         XLValue, Loc,
                         IsSeqCst ? llvm::AtomicOrdering::SequentiallyConsistent
                                  : llvm::AtomicOrdering::Monotonic,
                         XLValue.isVolatile());
  // OpenMP, 2.12.6, atomic Construct
  // Any atomic construct with a seq_cst clause forces the atomically
  // performed operation to include an implicit flush operation without a
  // list.
  if (IsSeqCst)
    CGF.CGM.getOpenMPRuntime().emitFlush(CGF, llvm::None, Loc);
  CGF.emitOMPSimpleStore(VLValue, Res, X->getType().getNonReferenceType(), Loc);
}

static void EmitOMPAtomicWriteExpr(CodeGenFunction &CGF, bool IsSeqCst,
                                   const Expr *X, const Expr *E,
                                   SourceLocation Loc) {
  // x = expr;
  assert(X->isLValue() && "X of 'omp atomic write' is not lvalue");
  emitSimpleAtomicStore(CGF, IsSeqCst, CGF.EmitLValue(X), CGF.EmitAnyExpr(E));
  // OpenMP, 2.12.6, atomic Construct
  // Any atomic construct with a seq_cst clause forces the atomically
  // performed operation to include an implicit flush operation without a
  // list.
  if (IsSeqCst)
    CGF.CGM.getOpenMPRuntime().emitFlush(CGF, llvm::None, Loc);
}

static std::pair<bool, RValue> emitOMPAtomicRMW(CodeGenFunction &CGF, LValue X,
                                                RValue Update,
                                                BinaryOperatorKind BO,
                                                llvm::AtomicOrdering AO,
                                                bool IsXLHSInRHSPart) {
  //auto &Context = CGF.CGM.getContext();
  // Allow atomicrmw only if 'x' and 'update' are integer values, lvalue for 'x'
  // expression is simple and atomic is allowed for the given type for the
  // target platform.

  // since the OpenMP requires user to guarantee the alignment, the
  // compiler check for alignment is removed.
  // The alignment check is OK for c frontend, but may fail for Fortran
  // this change should not be merged into the trunk
  if (BO == BO_Comma || !Update.isScalar() ||
      !Update.getScalarVal()->getType()->isIntegerTy() ||
      !X.isSimple() || (!isa<llvm::ConstantInt>(Update.getScalarVal()) &&
                        (Update.getScalarVal()->getType() !=
                         X.getAddress().getElementType())) ||
      !X.getAddress().getElementType()->isIntegerTy() /*||
      !Context.getTargetInfo().hasBuiltinAtomic(
          Context.getTypeSize(X.getType()), Context.toBits(X.getAlignment()))*/)
    return std::make_pair(false, RValue::get(nullptr));

  llvm::AtomicRMWInst::BinOp RMWOp;
  switch (BO) {
  case BO_Add:
    RMWOp = llvm::AtomicRMWInst::Add;
    break;
  case BO_Sub:
    if (!IsXLHSInRHSPart)
      return std::make_pair(false, RValue::get(nullptr));
    RMWOp = llvm::AtomicRMWInst::Sub;
    break;
  case BO_And:
    RMWOp = llvm::AtomicRMWInst::And;
    break;
  case BO_Or:
    RMWOp = llvm::AtomicRMWInst::Or;
    break;
  case BO_Xor:
    RMWOp = llvm::AtomicRMWInst::Xor;
    break;
  case BO_LT:
    RMWOp = X.getType()->hasSignedIntegerRepresentation()
                ? (IsXLHSInRHSPart ? llvm::AtomicRMWInst::Min
                                   : llvm::AtomicRMWInst::Max)
                : (IsXLHSInRHSPart ? llvm::AtomicRMWInst::UMin
                                   : llvm::AtomicRMWInst::UMax);
    break;
  case BO_GT:
    RMWOp = X.getType()->hasSignedIntegerRepresentation()
                ? (IsXLHSInRHSPart ? llvm::AtomicRMWInst::Max
                                   : llvm::AtomicRMWInst::Min)
                : (IsXLHSInRHSPart ? llvm::AtomicRMWInst::UMax
                                   : llvm::AtomicRMWInst::UMin);
    break;
  case BO_Assign:
    RMWOp = llvm::AtomicRMWInst::Xchg;
    break;
  case BO_Mul:
  case BO_Div:
  case BO_Rem:
  case BO_Shl:
  case BO_Shr:
  case BO_LAnd:
  case BO_LOr:
    return std::make_pair(false, RValue::get(nullptr));
  case BO_PtrMemD:
  case BO_PtrMemI:
  case BO_LE:
  case BO_GE:
  case BO_EQ:
  case BO_NE:
  case BO_AddAssign:
  case BO_SubAssign:
  case BO_AndAssign:
  case BO_OrAssign:
  case BO_XorAssign:
  case BO_MulAssign:
  case BO_DivAssign:
  case BO_RemAssign:
  case BO_ShlAssign:
  case BO_ShrAssign:
  case BO_Comma:
    llvm_unreachable("Unsupported atomic update operation");
  }
  auto *UpdateVal = Update.getScalarVal();
  if (auto *IC = dyn_cast<llvm::ConstantInt>(UpdateVal)) {
    UpdateVal = CGF.Builder.CreateIntCast(
        IC, X.getAddress().getElementType(),
        X.getType()->hasSignedIntegerRepresentation());
  }
  auto *Res = CGF.Builder.CreateAtomicRMW(RMWOp, X.getPointer(), UpdateVal, AO);
  return std::make_pair(true, RValue::get(Res));
}

std::pair<bool, RValue> CodeGenFunction::EmitOMPAtomicSimpleUpdateExpr(
    LValue X, RValue E, BinaryOperatorKind BO, bool IsXLHSInRHSPart,
    llvm::AtomicOrdering AO, SourceLocation Loc,
    const llvm::function_ref<RValue(RValue)> &CommonGen) {
  // Update expressions are allowed to have the following forms:
  // x binop= expr; -> xrval + expr;
  // x++, ++x -> xrval + 1;
  // x--, --x -> xrval - 1;
  // x = x binop expr; -> xrval binop expr
  // x = expr Op x; - > expr binop xrval;
  auto Res = emitOMPAtomicRMW(*this, X, E, BO, AO, IsXLHSInRHSPart);
  if (!Res.first) {
    if (X.isGlobalReg()) {
      // Emit an update expression: 'xrval' binop 'expr' or 'expr' binop
      // 'xrval'.
      EmitStoreThroughLValue(CommonGen(EmitLoadOfLValue(X, Loc)), X);
    } else {
      // Perform compare-and-swap procedure.
      EmitAtomicUpdate(X, AO, CommonGen, X.getType().isVolatileQualified());
    }
  }
  return Res;
}

static void EmitOMPAtomicUpdateExpr(CodeGenFunction &CGF, bool IsSeqCst,
                                    const Expr *X, const Expr *E,
                                    const Expr *UE, bool IsXLHSInRHSPart,
                                    SourceLocation Loc) {
  assert(isa<BinaryOperator>(UE->IgnoreImpCasts()) &&
         "Update expr in 'atomic update' must be a binary operator.");
  auto *BOUE = cast<BinaryOperator>(UE->IgnoreImpCasts());
  // Update expressions are allowed to have the following forms:
  // x binop= expr; -> xrval + expr;
  // x++, ++x -> xrval + 1;
  // x--, --x -> xrval - 1;
  // x = x binop expr; -> xrval binop expr
  // x = expr Op x; - > expr binop xrval;
  assert(X->isLValue() && "X of 'omp atomic update' is not lvalue");
  LValue XLValue = CGF.EmitLValue(X);
  RValue ExprRValue = CGF.EmitAnyExpr(E);
  auto AO = IsSeqCst ? llvm::AtomicOrdering::SequentiallyConsistent
                     : llvm::AtomicOrdering::Monotonic;
  auto *LHS = cast<OpaqueValueExpr>(BOUE->getLHS()->IgnoreImpCasts());
  auto *RHS = cast<OpaqueValueExpr>(BOUE->getRHS()->IgnoreImpCasts());
  auto *XRValExpr = IsXLHSInRHSPart ? LHS : RHS;
  auto *ERValExpr = IsXLHSInRHSPart ? RHS : LHS;
  auto Gen =
      [&CGF, UE, ExprRValue, XRValExpr, ERValExpr](RValue XRValue) -> RValue {
        CodeGenFunction::OpaqueValueMapping MapExpr(CGF, ERValExpr, ExprRValue);
        CodeGenFunction::OpaqueValueMapping MapX(CGF, XRValExpr, XRValue);
        return CGF.EmitAnyExpr(UE);
      };
  (void)CGF.EmitOMPAtomicSimpleUpdateExpr(
      XLValue, ExprRValue, BOUE->getOpcode(), IsXLHSInRHSPart, AO, Loc, Gen);
  // OpenMP, 2.12.6, atomic Construct
  // Any atomic construct with a seq_cst clause forces the atomically
  // performed operation to include an implicit flush operation without a
  // list.
  if (IsSeqCst)
    CGF.CGM.getOpenMPRuntime().emitFlush(CGF, llvm::None, Loc);
}

static RValue convertToType(CodeGenFunction &CGF, RValue Value,
                            QualType SourceType, QualType ResType,
                            SourceLocation Loc) {
  switch (CGF.getEvaluationKind(ResType)) {
  case TEK_Scalar:
    return RValue::get(
        convertToScalarValue(CGF, Value, SourceType, ResType, Loc));
  case TEK_Complex: {
    auto Res = convertToComplexValue(CGF, Value, SourceType, ResType, Loc);
    return RValue::getComplex(Res.first, Res.second);
  }
  case TEK_Aggregate:
    break;
  }
  llvm_unreachable("Must be a scalar or complex.");
}

static void EmitOMPAtomicCaptureExpr(CodeGenFunction &CGF, bool IsSeqCst,
                                     bool IsPostfixUpdate, const Expr *V,
                                     const Expr *X, const Expr *E,
                                     const Expr *UE, bool IsXLHSInRHSPart,
                                     SourceLocation Loc) {
  assert(X->isLValue() && "X of 'omp atomic capture' is not lvalue");
  assert(V->isLValue() && "V of 'omp atomic capture' is not lvalue");
  RValue NewVVal;
  LValue VLValue = CGF.EmitLValue(V);
  LValue XLValue = CGF.EmitLValue(X);
  RValue ExprRValue = CGF.EmitAnyExpr(E);
  auto AO = IsSeqCst ? llvm::AtomicOrdering::SequentiallyConsistent
                     : llvm::AtomicOrdering::Monotonic;
  QualType NewVValType;
  if (UE) {
    // 'x' is updated with some additional value.
    assert(isa<BinaryOperator>(UE->IgnoreImpCasts()) &&
           "Update expr in 'atomic capture' must be a binary operator.");
    auto *BOUE = cast<BinaryOperator>(UE->IgnoreImpCasts());
    // Update expressions are allowed to have the following forms:
    // x binop= expr; -> xrval + expr;
    // x++, ++x -> xrval + 1;
    // x--, --x -> xrval - 1;
    // x = x binop expr; -> xrval binop expr
    // x = expr Op x; - > expr binop xrval;
    auto *LHS = cast<OpaqueValueExpr>(BOUE->getLHS()->IgnoreImpCasts());
    auto *RHS = cast<OpaqueValueExpr>(BOUE->getRHS()->IgnoreImpCasts());
    auto *XRValExpr = IsXLHSInRHSPart ? LHS : RHS;
    NewVValType = XRValExpr->getType();
    auto *ERValExpr = IsXLHSInRHSPart ? RHS : LHS;
    auto &&Gen = [&CGF, &NewVVal, UE, ExprRValue, XRValExpr, ERValExpr,
                  IsSeqCst, IsPostfixUpdate](RValue XRValue) -> RValue {
      CodeGenFunction::OpaqueValueMapping MapExpr(CGF, ERValExpr, ExprRValue);
      CodeGenFunction::OpaqueValueMapping MapX(CGF, XRValExpr, XRValue);
      RValue Res = CGF.EmitAnyExpr(UE);
      NewVVal = IsPostfixUpdate ? XRValue : Res;
      return Res;
    };
    auto Res = CGF.EmitOMPAtomicSimpleUpdateExpr(
        XLValue, ExprRValue, BOUE->getOpcode(), IsXLHSInRHSPart, AO, Loc, Gen);
    if (Res.first) {
      // 'atomicrmw' instruction was generated.
      if (IsPostfixUpdate) {
        // Use old value from 'atomicrmw'.
        NewVVal = Res.second;
      } else {
        // 'atomicrmw' does not provide new value, so evaluate it using old
        // value of 'x'.
        CodeGenFunction::OpaqueValueMapping MapExpr(CGF, ERValExpr, ExprRValue);
        CodeGenFunction::OpaqueValueMapping MapX(CGF, XRValExpr, Res.second);
        NewVVal = CGF.EmitAnyExpr(UE);
      }
    }
  } else {
    // 'x' is simply rewritten with some 'expr'.
    NewVValType = X->getType().getNonReferenceType();
    ExprRValue = convertToType(CGF, ExprRValue, E->getType(),
                               X->getType().getNonReferenceType(), Loc);
    auto &&Gen = [&CGF, &NewVVal, ExprRValue](RValue XRValue) -> RValue {
      NewVVal = XRValue;
      return ExprRValue;
    };
    // Try to perform atomicrmw xchg, otherwise simple exchange.
    auto Res = CGF.EmitOMPAtomicSimpleUpdateExpr(
        XLValue, ExprRValue, /*BO=*/BO_Assign, /*IsXLHSInRHSPart=*/false, AO,
        Loc, Gen);
    if (Res.first) {
      // 'atomicrmw' instruction was generated.
      NewVVal = IsPostfixUpdate ? Res.second : ExprRValue;
    }
  }
  // Emit post-update store to 'v' of old/new 'x' value.
  CGF.emitOMPSimpleStore(VLValue, NewVVal, NewVValType, Loc);
  // OpenMP, 2.12.6, atomic Construct
  // Any atomic construct with a seq_cst clause forces the atomically
  // performed operation to include an implicit flush operation without a
  // list.
  if (IsSeqCst)
    CGF.CGM.getOpenMPRuntime().emitFlush(CGF, llvm::None, Loc);
}

static void EmitOMPAtomicExpr(CodeGenFunction &CGF, OpenMPClauseKind Kind,
                              bool IsSeqCst, bool IsPostfixUpdate,
                              const Expr *X, const Expr *V, const Expr *E,
                              const Expr *UE, bool IsXLHSInRHSPart,
                              SourceLocation Loc) {
  switch (Kind) {
  case OMPC_read:
    EmitOMPAtomicReadExpr(CGF, IsSeqCst, X, V, Loc);
    break;
  case OMPC_write:
    EmitOMPAtomicWriteExpr(CGF, IsSeqCst, X, E, Loc);
    break;
  case OMPC_unknown:
  case OMPC_update:
    EmitOMPAtomicUpdateExpr(CGF, IsSeqCst, X, E, UE, IsXLHSInRHSPart, Loc);
    break;
  case OMPC_capture:
    EmitOMPAtomicCaptureExpr(CGF, IsSeqCst, IsPostfixUpdate, V, X, E, UE,
                             IsXLHSInRHSPart, Loc);
    break;
  case OMPC_if:
  case OMPC_final:
  case OMPC_num_threads:
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_lastprivate:
  case OMPC_reduction:
  case OMPC_safelen:
  case OMPC_simdlen:
  case OMPC_collapse:
  case OMPC_default:
  case OMPC_seq_cst:
  case OMPC_shared:
  case OMPC_linear:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_flush:
  case OMPC_lastprivate_update:
  case OMPC_proc_bind:
  case OMPC_schedule:
  case OMPC_ordered:
  case OMPC_nowait:
  case OMPC_untied:
  case OMPC_threadprivate:
  case OMPC_depend:
  case OMPC_mergeable:
  case OMPC_device:
  case OMPC_threads:
  case OMPC_simd:
  case OMPC_map:
  case OMPC_num_teams:
  case OMPC_thread_limit:
  case OMPC_priority:
  case OMPC_grainsize:
  case OMPC_nogroup:
  case OMPC_num_tasks:
  case OMPC_hint:
  case OMPC_dist_schedule:
  case OMPC_defaultmap:
  case OMPC_uniform:
  case OMPC_to:
  case OMPC_from:
  case OMPC_use_device_ptr:
  case OMPC_is_device_ptr:
  case OMPC_task_reduction:
  case OMPC_in_reduction:
    llvm_unreachable("Clause is not allowed in 'omp atomic'.");
  }
}

void CodeGenFunction::EmitOMPAtomicDirective(const OMPAtomicDirective &S) {
  bool IsSeqCst = S.getSingleClause<OMPSeqCstClause>();
  OpenMPClauseKind Kind = OMPC_unknown;
  for (auto *C : S.clauses()) {
    // Find first clause (skip seq_cst clause, if it is first).
    if (C->getClauseKind() != OMPC_seq_cst) {
      Kind = C->getClauseKind();
      break;
    }
  }

  const auto *CS =
      S.getAssociatedStmt()->IgnoreContainers(/*IgnoreCaptured=*/true);
  if (const auto *EWC = dyn_cast<ExprWithCleanups>(CS)) {
    enterFullExpression(EWC);
  }
  // Processing for statements under 'atomic capture'.
  if (const auto *Compound = dyn_cast<CompoundStmt>(CS)) {
    for (const auto *C : Compound->body()) {
      if (const auto *EWC = dyn_cast<ExprWithCleanups>(C)) {
        enterFullExpression(EWC);
      }
    }
  }

  auto &&CodeGen = [&S, Kind, IsSeqCst, CS](CodeGenFunction &CGF,
                                            PrePostActionTy &) {
    CGF.EmitStopPoint(CS);
    EmitOMPAtomicExpr(CGF, Kind, IsSeqCst, S.isPostfixUpdate(), S.getX(),
                      S.getV(), S.getExpr(), S.getUpdateExpr(),
                      S.isXLHSInRHSPart(), S.getLocStart());
  };
  OMPLexicalScope Scope(*this, S, /*AsInlined=*/true);
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_atomic, CodeGen);
}

static void emitCommonOMPTargetDirective(CodeGenFunction &CGF,
                                         const OMPExecutableDirective &S,
                                         OpenMPDirectiveKind InnermostKind,
                                         const RegionCodeGenTy &CodeGen,
                                         ArrayRef<llvm::Value *> CapturedVars,
                                         OMPMapArrays &MapArrays) {
  assert(isOpenMPTargetExecutionDirective(S.getDirectiveKind()));
  CodeGenModule &CGM = CGF.CGM;
  const auto *PCS = cast<CapturedStmt>(S.getAssociatedStmt());
  if (S.hasClausesOfKind<OMPDependClause>())
    PCS = cast<CapturedStmt>(PCS->getCapturedStmt());
  const CapturedStmt &CS = *PCS;

  // On device emit this construct as inlined code.
  if (CGM.getLangOpts().OpenMPIsDevice) {
    CGM.getOpenMPRuntime().emitInlinedDirective(
        CGF, OMPD_target, [&CS](CodeGenFunction &CGF, PrePostActionTy &) {
          CGF.EmitStmt(CS.getCapturedStmt());
        });
    return;
  }

  llvm::Function *Fn = nullptr;
  llvm::Constant *FnID = nullptr;

  // check if we have a depend clause
  bool HasDependClause = S.hasClausesOfKind<OMPDependClause>();

  // Check if we have any if clause associated with the directive.
  const Expr *IfCond = nullptr;

  for (const auto *C : S.getClausesOfKind<OMPIfClause>()) {
    if (C->getNameModifier() == OMPD_unknown ||
        C->getNameModifier() == OMPD_target) {
      IfCond = C->getCondition();
      break;
    }
  }

  // Check if we have any device clause associated with the directive.
  const Expr *Device = nullptr;
  if (auto *C = S.getSingleClause<OMPDeviceClause>()) {
    Device = C->getDevice();
  }

  // Check if we have an if clause whose conditional always evaluates to false
  // or if we do not have any targets specified. If so the target region is not
  // an offload entry point.
  bool IsOffloadEntry = true;
  if (IfCond) {
    bool Val;
    if (CGF.ConstantFoldsToSimpleInteger(IfCond, Val) && !Val)
      IsOffloadEntry = false;
  }
  if (CGM.getLangOpts().OMPTargetTriples.empty())
    IsOffloadEntry = false;

  assert(CGF.CurFuncDecl && "No parent declaration for target region!");
  StringRef ParentName;
  // In case we have Ctors/Dtors we use the complete type variant to produce
  // the mangling of the device outlined kernel.
  if (auto *D = dyn_cast<CXXConstructorDecl>(CGF.CurFuncDecl))
    ParentName = CGM.getMangledName(GlobalDecl(D, Ctor_Complete));
  else if (auto *D = dyn_cast<CXXDestructorDecl>(CGF.CurFuncDecl))
    ParentName = CGM.getMangledName(GlobalDecl(D, Dtor_Complete));
  else
    ParentName =
        CGM.getMangledName(GlobalDecl(cast<FunctionDecl>(CGF.CurFuncDecl)));

  // Emit target region as a standalone region.
  unsigned CaptureLevel = HasDependClause ? 2 : 1;
  CGM.getOpenMPRuntime().emitTargetOutlinedFunction(
      S, ParentName, Fn, FnID, IsOffloadEntry, CodeGen, CaptureLevel);

  auto &&SizeEmitter = [](CodeGenFunction &CGF,
                          const OMPLoopDirective &D) -> llvm::Value * {
    OMPLoopScope(CGF, D, /*EmitPreInit=*/true);
    // Emit calculation of the iterations count.
    llvm::Value *NumIterations = CGF.EmitScalarExpr(D.getNumIterations());
    NumIterations = CGF.Builder.CreateIntCast(NumIterations, CGF.Int64Ty,
                                              /*IsSigned=*/false);
    return NumIterations;
  };
  CGM.getOpenMPRuntime().emitTargetNumIterationsCall(CGF, S, Device,
                                                     SizeEmitter);
  CGM.getOpenMPRuntime().emitTargetCall(CGF, S, Fn, FnID, IfCond, Device,
                                        CapturedVars, MapArrays);
}

void TargetCodegen(CodeGenFunction &CGF, PrePostActionTy &Action,
                   const OMPTargetDirective &S) {
  CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
  (void)CGF.EmitOMPFirstprivateClause(S, PrivateScope);
  CGF.EmitOMPPrivateClause(S, PrivateScope);
  (void)PrivateScope.Privatize();

  Action.Enter(CGF);
  const Stmt *CS = cast<CapturedStmt>(S.getAssociatedStmt())->getCapturedStmt();
  if (S.hasClausesOfKind<OMPDependClause>())
    CS = cast<CapturedStmt>(CS)->getCapturedStmt();
  CGF.EmitStmt(CS);
}

void CodeGenFunction::EmitOMPTargetDeviceFunction(CodeGenModule &CGM,
                                                  StringRef ParentName,
                                                  const OMPTargetDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetCodegen(CGF, Action, S);
  };
  llvm::Function *Fn;
  llvm::Constant *Addr;
  // Emit target region as a standalone region.
  CGM.getOpenMPRuntime().emitTargetOutlinedFunction(
      S, ParentName, Fn, Addr, /*IsOffloadEntry=*/true, CodeGen,
      /* CaptureLevel = */ 1);
  assert(Fn && Addr && "Target device function emission failed.");
}

static void
generateCapturedVarsAndMapArrays(CodeGenFunction &CGF,
                                 const OMPExecutableDirective &S,
                                 SmallVectorImpl<llvm::Value *> &CapturedVars,
                                 OMPMapArrays &MapArrays, bool HasDepend) {
  const auto *C = cast<CapturedStmt>(S.getAssociatedStmt());
  if (HasDepend)
    C = cast<CapturedStmt>(C->getCapturedStmt());
  const CapturedStmt &CS = *C;
  CGF.GenerateOpenMPCapturedVars(CS, CapturedVars);
  MapArrays =
      CGF.CGM.getOpenMPRuntime().generateMapArrays(CGF, S, CapturedVars);
}

void CodeGenFunction::EmitOMPTargetDirective(const OMPTargetDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetCodegen(CGF, Action, S);
  };
  llvm::SmallVector<llvm::Value *, 16> CapturedVars;
  OMPMapArrays MapArrays;
  bool HasDepend = S.hasClausesOfKind<OMPDependClause>();
  OMPLexicalScope TaskScope(*this, S, true);
  auto &&ArgGen = [&S, &CapturedVars, &MapArrays,
                   HasDepend](CodeGenFunction &CGF, PrePostActionTy &) {
    generateCapturedVarsAndMapArrays(CGF, S, CapturedVars, MapArrays,
                                     HasDepend);
  };
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_target, ArgGen);
  emitCommonOMPTargetDirective(*this, S, OMPD_target, CodeGen, CapturedVars,
                               MapArrays);
}

static void TargetParallelCodegen(CodeGenFunction &CGF, PrePostActionTy &Action,
                                  const OMPTargetParallelDirective &S) {
  Action.Enter(CGF);
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
    (void)CGF.EmitOMPFirstprivateClause(S, PrivateScope);
    CGF.EmitOMPPrivateClause(S, PrivateScope);
    CGF.EmitOMPReductionClauseInit(S, PrivateScope);
    (void)PrivateScope.Privatize();
    const auto *CS = cast<CapturedStmt>(S.getAssociatedStmt());
    if (S.hasClausesOfKind<OMPDependClause>())
      CS = cast<CapturedStmt>(CS->getCapturedStmt());
    CGF.EmitStmt(CS->getCapturedStmt());
    CGF.EmitOMPReductionClauseFinal(S, OMPD_parallel);
  };
  emitCommonOMPParallelDirective(CGF, S, OMPD_parallel, CodeGen,
                                 /*CaptureLevel=*/2);
  emitPostUpdateForReductionClause(CGF, S,
                                   [](CodeGenFunction &) { return nullptr; });
}

void CodeGenFunction::EmitOMPTargetParallelDeviceFunction(
    CodeGenModule &CGM, StringRef ParentName,
    const OMPTargetParallelDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetParallelCodegen(CGF, Action, S);
  };
  llvm::Function *Fn;
  llvm::Constant *Addr;
  // Emit target region as a standalone region.
  CGM.getOpenMPRuntime().emitTargetOutlinedFunction(
      S, ParentName, Fn, Addr, /*IsOffloadEntry=*/true, CodeGen,
      /* CaptureLevel = */ 1);
  assert(Fn && Addr && "Target device function emission failed.");
}

void CodeGenFunction::EmitOMPTargetParallelDirective(
    const OMPTargetParallelDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetParallelCodegen(CGF, Action, S);
  };
  OMPLexicalScope TaskScope(*this, S, true);
  llvm::SmallVector<llvm::Value *, 16> CapturedVars;
  OMPMapArrays MapArrays;
  bool HasDepend = S.hasClausesOfKind<OMPDependClause>();
  auto &&ArgGen = [&S, &CapturedVars, &MapArrays,
                   HasDepend](CodeGenFunction &CGF, PrePostActionTy &) {
    generateCapturedVarsAndMapArrays(CGF, S, CapturedVars, MapArrays,
                                     HasDepend);
  };
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_target, ArgGen);
  emitCommonOMPTargetDirective(*this, S, OMPD_target_parallel, CodeGen,
                               CapturedVars, MapArrays);
}

static void TargetSimdCodegen(CodeGenFunction &CGF, PrePostActionTy &Action,
                              const OMPTargetSimdDirective &S) {
  CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
  (void)CGF.EmitOMPFirstprivateClause(S, PrivateScope);
  CGF.EmitOMPPrivateClause(S, PrivateScope);
  (void)PrivateScope.Privatize();

  Action.Enter(CGF);
  CGF.EmitOMPSimdLoop(S, false);
}

void CodeGenFunction::EmitOMPTargetSimdDirective(
    const OMPTargetSimdDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetSimdCodegen(CGF, Action, S);
  };
  OMPLexicalScope TaskScope(*this, S, true);
  llvm::SmallVector<llvm::Value *, 16> CapturedVars;
  OMPMapArrays MapArrays;
  bool HasDepend = S.hasClausesOfKind<OMPDependClause>();
  auto &&ArgGen = [&S, &CapturedVars, &MapArrays,
                   HasDepend](CodeGenFunction &CGF, PrePostActionTy &) {
    generateCapturedVarsAndMapArrays(CGF, S, CapturedVars, MapArrays,
                                     HasDepend);
  };
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_target, ArgGen);
  emitCommonOMPTargetDirective(*this, S, OMPD_target_simd, CodeGen,
                               CapturedVars, MapArrays);
}

void CodeGenFunction::EmitOMPTargetSimdDeviceFunction(
    CodeGenModule &CGM, StringRef ParentName, const OMPTargetSimdDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetSimdCodegen(CGF, Action, S);
  };
  llvm::Function *Fn;
  llvm::Constant *Addr;
  // Emit target region as a standalone region.
  CGM.getOpenMPRuntime().emitTargetOutlinedFunction(
      S, ParentName, Fn, Addr, /*IsOffloadEntry=*/true, CodeGen,
      /* CaptureLevel = */ 1);
  assert(Fn && Addr && "Target device function emission failed.");
}

static void TargetParallelForCodegen(CodeGenFunction &CGF,
                                     PrePostActionTy &Action,
                                     const OMPTargetParallelForDirective &S) {
  Action.Enter(CGF);
  // Emit directive as a combined directive that consists of two implicit
  // directives: 'parallel' with 'for' directive.
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    CodeGenFunction::OMPCancelStackRAII CancelRegion(
        CGF, OMPD_target_parallel_for, S.hasCancel());
    CGF.EmitOMPWorksharingLoop(S);
  };
  emitCommonOMPParallelDirective(CGF, S, OMPD_parallel_for, CodeGen,
                                 /*CaptureLevel=*/2);
}

void CodeGenFunction::EmitOMPTargetParallelForDeviceFunction(
    CodeGenModule &CGM, StringRef ParentName,
    const OMPTargetParallelForDirective &S) {
  // Emit SPMD target parallel for region as a standalone region.
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetParallelForCodegen(CGF, Action, S);
  };
  llvm::Function *Fn;
  llvm::Constant *Addr;
  // Emit target region as a standalone region.
  CGM.getOpenMPRuntime().emitTargetOutlinedFunction(
      S, ParentName, Fn, Addr, /*IsOffloadEntry=*/true, CodeGen,
      /* CaptureLevel = */ 1);
  assert(Fn && Addr && "Target device function emission failed.");
}

void CodeGenFunction::EmitOMPTargetParallelForDirective(
    const OMPTargetParallelForDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetParallelForCodegen(CGF, Action, S);
  };
  OMPLexicalScope TaskScope(*this, S, true);
  llvm::SmallVector<llvm::Value *, 16> CapturedVars;
  OMPMapArrays MapArrays;
  bool HasDepend = S.hasClausesOfKind<OMPDependClause>();
  auto &&ArgGen = [&S, &CapturedVars, &MapArrays,
                   HasDepend](CodeGenFunction &CGF, PrePostActionTy &) {
    generateCapturedVarsAndMapArrays(CGF, S, CapturedVars, MapArrays,
                                     HasDepend);
  };
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_target, ArgGen);
  emitCommonOMPTargetDirective(*this, S, OMPD_target_parallel_for, CodeGen,
                               CapturedVars, MapArrays);
}

static void
TargetParallelForSimdCodegen(CodeGenFunction &CGF, PrePostActionTy &Action,
                             const OMPTargetParallelForSimdDirective &S) {
  Action.Enter(CGF);
  // Emit directive as a combined directive that consists of two implicit
  // directives: 'parallel' with 'for' directive.
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    CGF.EmitOMPWorksharingLoop(S);
  };
  emitCommonOMPParallelDirective(CGF, S, OMPD_parallel_for_simd, CodeGen,
                                 /*CaptureLevel=*/2);
}

void CodeGenFunction::EmitOMPTargetParallelForSimdDeviceFunction(
    CodeGenModule &CGM, StringRef ParentName,
    const OMPTargetParallelForSimdDirective &S) {
  // Emit SPMD target parallel for region as a standalone region.
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetParallelForSimdCodegen(CGF, Action, S);
  };
  llvm::Function *Fn;
  llvm::Constant *Addr;
  // Emit target region as a standalone region.
  CGM.getOpenMPRuntime().emitTargetOutlinedFunction(
      S, ParentName, Fn, Addr, /*IsOffloadEntry=*/true, CodeGen,
      /* CaptureLevel = */ 1);
  assert(Fn && Addr && "Target device function emission failed.");
}

void CodeGenFunction::EmitOMPTargetParallelForSimdDirective(
    const OMPTargetParallelForSimdDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetParallelForSimdCodegen(CGF, Action, S);
  };
  OMPLexicalScope TaskScope(*this, S, true);
  llvm::SmallVector<llvm::Value *, 16> CapturedVars;
  OMPMapArrays MapArrays;
  bool HasDepend = S.hasClausesOfKind<OMPDependClause>();
  auto &&ArgGen = [&S, &CapturedVars, &MapArrays,
                   HasDepend](CodeGenFunction &CGF, PrePostActionTy &) {
    generateCapturedVarsAndMapArrays(CGF, S, CapturedVars, MapArrays,
                                     HasDepend);
  };
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_target, ArgGen);
  emitCommonOMPTargetDirective(*this, S, OMPD_target_parallel_for_simd, CodeGen,
                               CapturedVars, MapArrays);
}

static void TargetTeamsDistributeParallelForCodegen(
    CodeGenModule &CGM, CodeGenFunction &CGF, PrePostActionTy &Action,
    const OMPTargetTeamsDistributeParallelForDirective &S) {
  Action.Enter(CGF);
  auto &&CodeGen = [&CGM, &S](CodeGenFunction &CGF, PrePostActionTy &) {
    CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
    CGF.EmitOMPReductionClauseInit(S, PrivateScope);
    (void)PrivateScope.Privatize();
    auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
      auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
        auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
          CodeGenFunction::OMPCancelStackRAII CancelRegion(
              CGF, S.getDirectiveKind(), S.hasCancel());
          CGF.EmitOMPWorksharingLoop(S);
        };
        emitCommonOMPParallelDirective(CGF, S, OMPD_for, CodeGen);
      };
      CGF.EmitOMPDistributeLoop(S, CodeGen);
    };
    CGM.getOpenMPRuntime().emitInlinedDirective(CGF, OMPD_distribute, CodeGen,
                                                false);
    CGF.EmitOMPReductionClauseFinal(S, OMPD_teams);
  };
  emitCommonOMPTeamsDirective(CGF, S, OMPD_target_teams_distribute_parallel_for,
                              CodeGen, /*CaptureLevel=*/1,
                              /*ImplicitParamStop=*/2);
  emitPostUpdateForReductionClause(
      CGF, S, [](CodeGenFunction &) -> llvm::Value * { return nullptr; });
}

void CodeGenFunction::EmitOMPTargetTeamsDistributeParallelForDeviceFunction(
    CodeGenModule &CGM, StringRef ParentName,
    const OMPTargetTeamsDistributeParallelForDirective &S) {
  // Emit SPMD target parallel for region as a standalone region.
  auto &&CodeGen = [&S, &CGM](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetTeamsDistributeParallelForCodegen(CGM, CGF, Action, S);
  };
  llvm::Function *Fn;
  llvm::Constant *Addr;
  // Emit target region as a standalone region.
  CGM.getOpenMPRuntime().emitTargetOutlinedFunction(
      S, ParentName, Fn, Addr, /*IsOffloadEntry=*/true, CodeGen,
      /* CaptureLevel = */ 1);
  assert(Fn && Addr && "Target device function emission failed.");
}

void CodeGenFunction::EmitOMPTargetTeamsDistributeParallelForDirective(
    const OMPTargetTeamsDistributeParallelForDirective &S) {
  auto &&CodeGen = [&S, this](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetTeamsDistributeParallelForCodegen(CGM, CGF, Action, S);
  };
  OMPLexicalScope TaskScope(*this, S, true);
  llvm::SmallVector<llvm::Value *, 16> CapturedVars;
  OMPMapArrays MapArrays;
  bool HasDepend = S.hasClausesOfKind<OMPDependClause>();
  auto &&ArgGen = [&S, &CapturedVars, &MapArrays,
                   HasDepend](CodeGenFunction &CGF, PrePostActionTy &) {
    generateCapturedVarsAndMapArrays(CGF, S, CapturedVars, MapArrays,
                                     HasDepend);
  };
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_target, ArgGen);
  emitCommonOMPTargetDirective(*this, S,
                               OMPD_target_teams_distribute_parallel_for,
                               CodeGen, CapturedVars, MapArrays);
}

static void TargetTeamsDistributeParallelForSimdCodegen(
    CodeGenModule &CGM, CodeGenFunction &CGF, PrePostActionTy &Action,
    const OMPTargetTeamsDistributeParallelForSimdDirective &S) {
  Action.Enter(CGF);
  auto &&CodeGen = [&CGM, &S](CodeGenFunction &CGF, PrePostActionTy &) {
    CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
    CGF.EmitOMPReductionClauseInit(S, PrivateScope);
    (void)PrivateScope.Privatize();
    auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
      auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
        auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
          CGF.EmitOMPWorksharingLoop(S);
        };
        emitCommonOMPParallelDirective(CGF, S, OMPD_for, CodeGen);
      };
      CGF.EmitOMPDistributeLoop(S, CodeGen);
    };
    CGM.getOpenMPRuntime().emitInlinedDirective(CGF, OMPD_distribute, CodeGen,
                                                false);
    CGF.EmitOMPReductionClauseFinal(S, OMPD_teams);
  };
  emitCommonOMPTeamsDirective(CGF, S,
                              OMPD_target_teams_distribute_parallel_for_simd,
                              CodeGen, /*CaptureLevel=*/1,
                              /*ImplicitParamStop=*/2);
  emitPostUpdateForReductionClause(
      CGF, S, [](CodeGenFunction &) -> llvm::Value * { return nullptr; });
}

void CodeGenFunction::EmitOMPTargetTeamsDistributeParallelForSimdDeviceFunction(
    CodeGenModule &CGM, StringRef ParentName,
    const OMPTargetTeamsDistributeParallelForSimdDirective &S) {
  // Emit SPMD target parallel for region as a standalone region.
  auto &&CodeGen = [&S, &CGM](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetTeamsDistributeParallelForSimdCodegen(CGM, CGF, Action, S);
  };
  llvm::Function *Fn;
  llvm::Constant *Addr;
  // Emit target region as a standalone region.
  CGM.getOpenMPRuntime().emitTargetOutlinedFunction(
      S, ParentName, Fn, Addr, /*IsOffloadEntry=*/true, CodeGen,
      /* CaptureLevel = */ 1);
  assert(Fn && Addr && "Target device function emission failed.");
}

void CodeGenFunction::EmitOMPTargetTeamsDistributeParallelForSimdDirective(
    const OMPTargetTeamsDistributeParallelForSimdDirective &S) {
  auto &&CodeGen = [&S, this](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetTeamsDistributeParallelForSimdCodegen(CGM, CGF, Action, S);
  };
  OMPLexicalScope TaskScope(*this, S, true);
  llvm::SmallVector<llvm::Value *, 16> CapturedVars;
  OMPMapArrays MapArrays;
  bool HasDepend = S.hasClausesOfKind<OMPDependClause>();
  auto &&ArgGen = [&S, &CapturedVars, &MapArrays,
                   HasDepend](CodeGenFunction &CGF, PrePostActionTy &) {
    generateCapturedVarsAndMapArrays(CGF, S, CapturedVars, MapArrays,
                                     HasDepend);
  };
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_target, ArgGen);
  emitCommonOMPTargetDirective(*this, S,
                               OMPD_target_teams_distribute_parallel_for_simd,
                               CodeGen, CapturedVars, MapArrays);
}

void CodeGenFunction::EmitOMPTeamsDirective(const OMPTeamsDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    OMPPrivateScope PrivateScope(CGF);
    (void)CGF.EmitOMPFirstprivateClause(S, PrivateScope);
    CGF.EmitOMPPrivateClause(S, PrivateScope);
    CGF.EmitOMPReductionClauseInit(S, PrivateScope);
    CGF.CGM.getOpenMPRuntime().emitInitDSBlock(CGF);
    (void)PrivateScope.Privatize();
    CGF.EmitStmt(cast<CapturedStmt>(S.getAssociatedStmt())->getCapturedStmt());
    CGF.EmitOMPReductionClauseFinal(S, OMPD_teams);
  };
  emitCommonOMPTeamsDirective(*this, S, OMPD_teams, CodeGen);
  emitPostUpdateForReductionClause(
      *this, S, [](CodeGenFunction &) -> llvm::Value * { return nullptr; });
}

static void TargetTeamsCodegen(CodeGenFunction &CGF, PrePostActionTy &Action,
                               const OMPTargetTeamsDirective &S) {
  Action.Enter(CGF);
  auto &&TeamsBodyCG = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
    (void)CGF.EmitOMPFirstprivateClause(S, PrivateScope);
    CGF.EmitOMPPrivateClause(S, PrivateScope);
    CGF.EmitOMPReductionClauseInit(S, PrivateScope);
    CGF.CGM.getOpenMPRuntime().emitInitDSBlock(CGF);
    (void)PrivateScope.Privatize();
    const auto *CS = cast<CapturedStmt>(S.getAssociatedStmt());
    if (S.hasClausesOfKind<OMPDependClause>())
      CS = cast<CapturedStmt>(CS->getCapturedStmt());
    CGF.EmitStmt(CS->getCapturedStmt());
    CGF.EmitOMPReductionClauseFinal(S, OMPD_teams);
  };
  emitCommonOMPTeamsDirective(CGF, S, OMPD_teams, TeamsBodyCG);
  emitPostUpdateForReductionClause(CGF, S,
                                   [](CodeGenFunction &) { return nullptr; });
}

void CodeGenFunction::EmitOMPTargetTeamsDeviceFunction(
    CodeGenModule &CGM, StringRef ParentName,
    const OMPTargetTeamsDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetTeamsCodegen(CGF, Action, S);
  };
  llvm::Function *Fn;
  llvm::Constant *Addr;
  // Emit target region as a standalone region.
  CGM.getOpenMPRuntime().emitTargetOutlinedFunction(
      S, ParentName, Fn, Addr, /*IsOffloadEntry=*/true, CodeGen,
      /* CaptureLevel = */ 1);
  assert(Fn && Addr && "Target device function emission failed.");
}

void CodeGenFunction::EmitOMPTargetTeamsDirective(
    const OMPTargetTeamsDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetTeamsCodegen(CGF, Action, S);
  };
  OMPLexicalScope TaskScope(*this, S, true);
  llvm::SmallVector<llvm::Value *, 16> CapturedVars;
  OMPMapArrays MapArrays;
  bool HasDepend = S.hasClausesOfKind<OMPDependClause>();
  auto &&ArgGen = [&S, &CapturedVars, &MapArrays,
                   HasDepend](CodeGenFunction &CGF, PrePostActionTy &) {
    generateCapturedVarsAndMapArrays(CGF, S, CapturedVars, MapArrays,
                                     HasDepend);
  };
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_target, ArgGen);
  emitCommonOMPTargetDirective(*this, S, OMPD_target, CodeGen, CapturedVars,
                               MapArrays);
}

static void
TargetTeamsDistributeCodegen(CodeGenFunction &CGF, PrePostActionTy &Action,
                             const OMPTargetTeamsDistributeDirective &S) {
  Action.Enter(CGF);
  auto &&CGDistributeInlined = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
    CGF.EmitOMPReductionClauseInit(S, PrivateScope);
    // TODO: Do we need this here?
    CGF.CGM.getOpenMPRuntime().emitInitDSBlock(CGF);
    (void)PrivateScope.Privatize();
    auto &&CGDistributeLoop = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
      CGF.EmitOMPDistributeLoop(S, [](CodeGenFunction &, PrePostActionTy &) {});
    };
    CGF.CGM.getOpenMPRuntime().emitInlinedDirective(CGF, OMPD_distribute,
                                                    CGDistributeLoop,
                                                    /*HasCancel=*/false);
    CGF.EmitOMPReductionClauseFinal(S, OMPD_teams);
  };
  emitCommonOMPTeamsDirective(CGF, S, OMPD_teams_distribute,
                              CGDistributeInlined, 1, 2);
  emitPostUpdateForReductionClause(
      CGF, S, [](CodeGenFunction &) -> llvm::Value * { return nullptr; });
}

void CodeGenFunction::EmitOMPTargetTeamsDistributeDeviceFunction(
    CodeGenModule &CGM, StringRef ParentName,
    const OMPTargetTeamsDistributeDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetTeamsDistributeCodegen(CGF, Action, S);
  };
  llvm::Function *Fn;
  llvm::Constant *Addr;
  // Emit target region as a standalone region.
  CGM.getOpenMPRuntime().emitTargetOutlinedFunction(
      S, ParentName, Fn, Addr, /*IsOffloadEntry=*/true, CodeGen,
      /* CaptureLevel = */ 1);
  assert(Fn && Addr && "Target device function emission failed.");
}

void CodeGenFunction::EmitOMPTargetTeamsDistributeDirective(
    const OMPTargetTeamsDistributeDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetTeamsDistributeCodegen(CGF, Action, S);
  };
  OMPLexicalScope TaskScope(*this, S, true);
  llvm::SmallVector<llvm::Value *, 16> CapturedVars;
  OMPMapArrays MapArrays;
  bool HasDepend = S.hasClausesOfKind<OMPDependClause>();
  auto &&ArgGen = [&S, &CapturedVars, &MapArrays,
                   HasDepend](CodeGenFunction &CGF, PrePostActionTy &) {
    generateCapturedVarsAndMapArrays(CGF, S, CapturedVars, MapArrays,
                                     HasDepend);
  };
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_target, ArgGen);
  emitCommonOMPTargetDirective(*this, S, OMPD_target, CodeGen, CapturedVars,
                               MapArrays);
}

static void TargetTeamsDistributeSimdCodegen(
    CodeGenFunction &CGF, PrePostActionTy &Action,
    const OMPTargetTeamsDistributeSimdDirective &S) {
  Action.Enter(CGF);
  auto &&CGDistributeInlined = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
    CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
    CGF.EmitOMPReductionClauseInit(S, PrivateScope);
    // TODO: Do we need this here?
    CGF.CGM.getOpenMPRuntime().emitInitDSBlock(CGF);
    (void)PrivateScope.Privatize();
    auto &&CGDistributeLoop = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
      auto &&CGSimd = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
        CGF.EmitOMPSimdLoop(S, false);
      };
      CGF.EmitOMPDistributeLoop(S, CGSimd);
    };
    CGF.CGM.getOpenMPRuntime().emitInlinedDirective(CGF, OMPD_distribute,
                                                    CGDistributeLoop,
                                                    /*HasCancel=*/false);
    CGF.EmitOMPReductionClauseFinal(S, OMPD_teams);
  };
  emitCommonOMPTeamsDirective(CGF, S, OMPD_teams_distribute_simd,
                              CGDistributeInlined, 1, 2);
  emitPostUpdateForReductionClause(
      CGF, S, [](CodeGenFunction &) -> llvm::Value * { return nullptr; });
}

void CodeGenFunction::EmitOMPTargetTeamsDistributeSimdDeviceFunction(
    CodeGenModule &CGM, StringRef ParentName,
    const OMPTargetTeamsDistributeSimdDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetTeamsDistributeSimdCodegen(CGF, Action, S);
  };
  llvm::Function *Fn;
  llvm::Constant *Addr;
  // Emit target region as a standalone region.
  CGM.getOpenMPRuntime().emitTargetOutlinedFunction(
      S, ParentName, Fn, Addr, /*IsOffloadEntry=*/true, CodeGen,
      /* CaptureLevel = */ 1);
  assert(Fn && Addr && "Target device function emission failed.");
}

void CodeGenFunction::EmitOMPTargetTeamsDistributeSimdDirective(
    const OMPTargetTeamsDistributeSimdDirective &S) {
  auto &&CodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &Action) {
    TargetTeamsDistributeSimdCodegen(CGF, Action, S);
  };
  OMPLexicalScope TaskScope(*this, S, true);
  llvm::SmallVector<llvm::Value *, 16> CapturedVars;
  OMPMapArrays MapArrays;
  bool HasDepend = S.hasClausesOfKind<OMPDependClause>();
  auto &&ArgGen = [&S, &CapturedVars, &MapArrays,
                   HasDepend](CodeGenFunction &CGF, PrePostActionTy &) {
    generateCapturedVarsAndMapArrays(CGF, S, CapturedVars, MapArrays,
                                     HasDepend);
  };
  CGM.getOpenMPRuntime().emitInlinedDirective(*this, OMPD_target, ArgGen);
  emitCommonOMPTargetDirective(*this, S, OMPD_target, CodeGen, CapturedVars,
                               MapArrays);
}

void CodeGenFunction::EmitOMPCancellationPointDirective(
    const OMPCancellationPointDirective &S) {
  CGM.getOpenMPRuntime().emitCancellationPointCall(*this, S.getLocStart(),
                                                   S.getCancelRegion());
}

void CodeGenFunction::EmitOMPCancelDirective(const OMPCancelDirective &S) {
  const Expr *IfCond = nullptr;
  for (const auto *C : S.getClausesOfKind<OMPIfClause>()) {
    if (C->getNameModifier() == OMPD_unknown ||
        C->getNameModifier() == OMPD_cancel) {
      IfCond = C->getCondition();
      break;
    }
  }
  CGM.getOpenMPRuntime().emitCancelCall(*this, S.getLocStart(), IfCond,
                                        S.getCancelRegion());
}

CodeGenFunction::JumpDest
CodeGenFunction::getOMPCancelDestination(OpenMPDirectiveKind Kind) {
  if (Kind == OMPD_parallel || Kind == OMPD_task ||
      Kind == OMPD_target_parallel)
    return ReturnBlock;
  assert(Kind == OMPD_for || Kind == OMPD_section || Kind == OMPD_sections ||
         Kind == OMPD_parallel_sections || Kind == OMPD_parallel_for ||
         Kind == OMPD_distribute_parallel_for ||
         Kind == OMPD_target_parallel_for ||
         Kind == OMPD_teams_distribute_parallel_for ||
         Kind == OMPD_target_teams_distribute_parallel_for);
  return OMPCancelStack.getExitBlock();
}

void CodeGenFunction::EmitOMPUseDevicePtrClause(
    const OMPClause &NC, OMPPrivateScope &PrivateScope,
    const llvm::DenseMap<const ValueDecl *, Address> &CaptureDeviceAddrMap) {
  const auto &C = cast<OMPUseDevicePtrClause>(NC);
  auto OrigVarIt = C.varlist_begin();
  auto InitIt = C.inits().begin();
  for (auto PvtVarIt : C.private_copies()) {
    auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(*OrigVarIt)->getDecl());
    auto *InitVD = cast<VarDecl>(cast<DeclRefExpr>(*InitIt)->getDecl());
    auto *PvtVD = cast<VarDecl>(cast<DeclRefExpr>(PvtVarIt)->getDecl());

    // In order to identify the right initializer we need to match the
    // declaration used by the mapping logic. In some cases we may get
    // OMPCapturedExprDecl that refers to the original declaration.
    const ValueDecl *MatchingVD = OrigVD;
    if (auto *OED = dyn_cast<OMPCapturedExprDecl>(MatchingVD)) {
      // OMPCapturedExprDecl are used to privative fields of the current
      // structure.
      auto *ME = cast<MemberExpr>(OED->getInit());
      assert(isa<CXXThisExpr>(ME->getBase()) &&
             "Base should be the current struct!");
      MatchingVD = ME->getMemberDecl();
    }

    // If we don't have information about the current list item, move on to
    // the next one.
    auto InitAddrIt = CaptureDeviceAddrMap.find(MatchingVD);
    if (InitAddrIt == CaptureDeviceAddrMap.end())
      continue;

    bool IsRegistered = PrivateScope.addPrivate(OrigVD, [&]() -> Address {
      // Initialize the temporary initialization variable with the address we
      // get from the runtime library. We have to cast the source address
      // because it is always a void *. References are materialized in the
      // privatization scope, so the initialization here disregards the fact
      // the original variable is a reference.
      QualType AddrQTy =
          getContext().getPointerType(OrigVD->getType().getNonReferenceType());
      llvm::Type *AddrTy = ConvertTypeForMem(AddrQTy);
      Address InitAddr = Builder.CreateBitCast(InitAddrIt->second, AddrTy);
      setAddrOfLocalVar(InitVD, InitAddr);

      // Emit private declaration, it will be initialized by the value we
      // declaration we just added to the local declarations map.
      EmitDecl(*PvtVD);

      // The initialization variables reached its purpose in the emission
      // ofthe previous declaration, so we don't need it anymore.
      LocalDeclMap.erase(InitVD);

      // Return the address of the private variable.
      return GetAddrOfLocalVar(PvtVD);
    });
    assert(IsRegistered && "firstprivate var already registered as private");
    // Silence the warning about unused variable.
    (void)IsRegistered;

    ++OrigVarIt;
    ++InitIt;
  }
}

// Generate the instructions for '#pragma omp target data' directive.
void CodeGenFunction::EmitOMPTargetDataDirective(
    const OMPTargetDataDirective &S) {
  CGOpenMPRuntime::TargetDataInfo Info(/*RequiresDevicePointerInfo=*/true);

  // Create a pre/post action to signal the privatization of the device pointer.
  // This action can be replaced by the OpenMP runtime code generation to
  // deactivate privatization.
  bool PrivatizeDevicePointers = false;
  class DevicePointerPrivActionTy : public PrePostActionTy {
    bool &PrivatizeDevicePointers;

  public:
    explicit DevicePointerPrivActionTy(bool &PrivatizeDevicePointers)
        : PrePostActionTy(), PrivatizeDevicePointers(PrivatizeDevicePointers) {}
    void Enter(CodeGenFunction &CGF) override {
      PrivatizeDevicePointers = true;
    }
  };
  DevicePointerPrivActionTy PrivAction(PrivatizeDevicePointers);

  auto &&CodeGen = [&S, &Info, &PrivatizeDevicePointers](
      CodeGenFunction &CGF, PrePostActionTy &Action) {
    auto &&InnermostCodeGen = [&S](CodeGenFunction &CGF, PrePostActionTy &) {
      CGF.EmitStmt(
          cast<CapturedStmt>(S.getAssociatedStmt())->getCapturedStmt());
    };

    // Codegen that selects wheather to generate the privatization code or not.
    auto &&PrivCodeGen = [&S, &Info, &PrivatizeDevicePointers,
                          &InnermostCodeGen](CodeGenFunction &CGF,
                                             PrePostActionTy &Action) {
      RegionCodeGenTy RCG(InnermostCodeGen);
      PrivatizeDevicePointers = false;

      // Call the pre-action to change the status of PrivatizeDevicePointers if
      // needed.
      Action.Enter(CGF);

      if (PrivatizeDevicePointers) {
        OMPPrivateScope PrivateScope(CGF);
        // Emit all instances of the use_device_ptr clause.
        for (const auto *C : S.getClausesOfKind<OMPUseDevicePtrClause>())
          CGF.EmitOMPUseDevicePtrClause(*C, PrivateScope,
                                        Info.CaptureDeviceAddrMap);
        (void)PrivateScope.Privatize();
        RCG(CGF);
      } else
        RCG(CGF);
    };

    // Forward the provided action to the privatization codegen.
    RegionCodeGenTy PrivRCG(PrivCodeGen);
    PrivRCG.setAction(Action);

    // Notwithstanding the body of the region is emitted as inlined directive,
    // we don't use an inline scope as changes in the references inside the
    // region are expected to be visible outside, so we do not privative them.
    OMPLexicalScope Scope(CGF, S);
    CGF.CGM.getOpenMPRuntime().emitInlinedDirective(CGF, OMPD_target_data,
                                                    PrivRCG);
  };

  RegionCodeGenTy RCG(CodeGen);

  // If we don't have target devices, don't bother emitting the data mapping
  // code.
  if (CGM.getLangOpts().OMPTargetTriples.empty()) {
    RCG(*this);
    return;
  }

  // Check if we have any if clause associated with the directive.
  const Expr *IfCond = nullptr;
  if (auto *C = S.getSingleClause<OMPIfClause>())
    IfCond = C->getCondition();

  // Check if we have any device clause associated with the directive.
  const Expr *Device = nullptr;
  if (auto *C = S.getSingleClause<OMPDeviceClause>())
    Device = C->getDevice();

  // Set the action to signal privatization of device pointers.
  RCG.setAction(PrivAction);

  // Emit region code.
  CGM.getOpenMPRuntime().emitTargetDataCalls(*this, S, IfCond, Device, RCG,
                                             Info);
}

void CodeGenFunction::EmitOMPTargetEnterDataDirective(
    const OMPTargetEnterDataDirective &S) {
  // If we don't have target devices, don't bother emitting the data mapping
  // code.
  if (CGM.getLangOpts().OMPTargetTriples.empty())
    return;

  // Check if we have any if clause associated with the directive.
  const Expr *IfCond = nullptr;
  if (auto *C = S.getSingleClause<OMPIfClause>())
    IfCond = C->getCondition();

  // Check if we have any device clause associated with the directive.
  const Expr *Device = nullptr;
  if (auto *C = S.getSingleClause<OMPDeviceClause>())
    Device = C->getDevice();

  CGM.getOpenMPRuntime().emitTargetDataStandAloneCall(*this, S, IfCond, Device);
}

void CodeGenFunction::EmitOMPTargetExitDataDirective(
    const OMPTargetExitDataDirective &S) {
  // If we don't have target devices, don't bother emitting the data mapping
  // code.
  if (CGM.getLangOpts().OMPTargetTriples.empty())
    return;

  // Check if we have any if clause associated with the directive.
  const Expr *IfCond = nullptr;
  if (auto *C = S.getSingleClause<OMPIfClause>())
    IfCond = C->getCondition();

  // Check if we have any device clause associated with the directive.
  const Expr *Device = nullptr;
  if (auto *C = S.getSingleClause<OMPDeviceClause>())
    Device = C->getDevice();

  CGM.getOpenMPRuntime().emitTargetDataStandAloneCall(*this, S, IfCond, Device);
}

/// Emit a helper variable and return corresponding lvalue.
static void mapParam(CodeGenFunction &CGF, const DeclRefExpr *Helper,
                     const ImplicitParamDecl *PVD,
                     CodeGenFunction::OMPPrivateScope &Privates) {
  auto *VDecl = cast<VarDecl>(Helper->getDecl());
  Privates.addPrivate(
      VDecl, [&CGF, PVD]() -> Address { return CGF.GetAddrOfLocalVar(PVD); });
}

void CodeGenFunction::EmitOMPTaskLoopBasedDirective(const OMPLoopDirective &S) {
  assert(isOpenMPTaskLoopDirective(S.getDirectiveKind()));
  // Emit outlined function for task construct.
  auto CS = cast<CapturedStmt>(S.getAssociatedStmt());
  auto CapturedStruct = GenerateCapturedStmtArgument(*CS);
  auto SharedsTy = getContext().getRecordType(CS->getCapturedRecordDecl());
  const Expr *IfCond = nullptr;
  for (const auto *C : S.getClausesOfKind<OMPIfClause>()) {
    if (C->getNameModifier() == OMPD_unknown ||
        C->getNameModifier() == OMPD_taskloop) {
      IfCond = C->getCondition();
      break;
    }
  }

  OMPTaskDataTy Data;
  // Check if taskloop must be emitted without taskgroup.
  Data.Nogroup = S.getSingleClause<OMPNogroupClause>();
  // TODO: Check if we should emit tied or untied task.
  Data.Tied = true;
  // Set scheduling for taskloop
  if (const auto* Clause = S.getSingleClause<OMPGrainsizeClause>()) {
    // grainsize clause
    Data.Schedule.setInt(/*IntVal=*/false);
    Data.Schedule.setPointer(EmitScalarExpr(Clause->getGrainsize()));
  } else if (const auto* Clause = S.getSingleClause<OMPNumTasksClause>()) {
    // num_tasks clause
    Data.Schedule.setInt(/*IntVal=*/true);
    Data.Schedule.setPointer(EmitScalarExpr(Clause->getNumTasks()));
  }

  auto &&BodyGen = [CS, &S](CodeGenFunction &CGF, PrePostActionTy &) {
    // if (PreCond) {
    //   for (IV in 0..LastIteration) BODY;
    //   <Final counter/linear vars updates>;
    // }
    //

    // Emit: if (PreCond) - begin.
    // If the condition constant folds and can be elided, avoid emitting the
    // whole loop.
    bool CondConstant;
    llvm::BasicBlock *ContBlock = nullptr;
    OMPLoopScope PreInitScope(CGF, S, /*EmitPreInit=*/true);
    if (CGF.ConstantFoldsToSimpleInteger(S.getPreCond(), CondConstant)) {
      if (!CondConstant)
        return;
    } else {
      auto *ThenBlock = CGF.createBasicBlock("taskloop.if.then");
      ContBlock = CGF.createBasicBlock("taskloop.if.end");
      emitPreCond(CGF, S, S.getPreCond(), ThenBlock, ContBlock,
                  CGF.getProfileCount(&S));
      CGF.EmitBlock(ThenBlock);
      CGF.incrementProfileCounter(&S);
    }

    if (isOpenMPSimdDirective(S.getDirectiveKind()))
      CGF.EmitOMPSimdInit(S);

    OMPPrivateScope LoopScope(CGF);
    // Emit helper vars inits.
    enum { LowerBound = 5, UpperBound, Stride, LastIter };
    auto *I = CS->getCapturedDecl()->param_begin();
    auto *LBP = std::next(I, LowerBound);
    auto *UBP = std::next(I, UpperBound);
    auto *STP = std::next(I, Stride);
    auto *LIP = std::next(I, LastIter);
    mapParam(CGF, cast<DeclRefExpr>(S.getLowerBoundVariable()), *LBP,
             LoopScope);
    mapParam(CGF, cast<DeclRefExpr>(S.getUpperBoundVariable()), *UBP,
             LoopScope);
    mapParam(CGF, cast<DeclRefExpr>(S.getStrideVariable()), *STP, LoopScope);
    mapParam(CGF, cast<DeclRefExpr>(S.getIsLastIterVariable()), *LIP,
             LoopScope);
    CGF.EmitOMPPrivateLoopCounters(S, LoopScope);
    bool HasLastprivateClause = CGF.EmitOMPLastprivateClauseInit(S, LoopScope);
    (void)LoopScope.Privatize();
    // Emit the loop iteration variable.
    const Expr *IVExpr = S.getIterationVariable();
    const VarDecl *IVDecl = cast<VarDecl>(cast<DeclRefExpr>(IVExpr)->getDecl());
    CGF.EmitVarDecl(*IVDecl);
    CGF.EmitIgnoredExpr(S.getInit());

    // Emit the iterations count variable.
    // If it is not a variable, Sema decided to calculate iterations count on
    // each iteration (e.g., it is foldable into a constant).
    if (auto LIExpr = dyn_cast<DeclRefExpr>(S.getLastIteration())) {
      CGF.EmitVarDecl(*cast<VarDecl>(LIExpr->getDecl()));
      // Emit calculation of the iterations count.
      CGF.EmitIgnoredExpr(S.getCalcLastIteration());
    }

    CGF.EmitOMPInnerLoop(S, LoopScope.requiresCleanups(), S.getCond(),
                         S.getInc(),
                         /*LastprivateIterInitExpr=*/nullptr,
                         [&S](CodeGenFunction &CGF) {
                           CGF.EmitOMPLoopBody(S, JumpDest());
                           CGF.EmitStopPoint(&S);
                         },
                         [](CodeGenFunction &) {});
    // Emit: if (PreCond) - end.
    if (ContBlock) {
      CGF.EmitBranch(ContBlock);
      CGF.EmitBlock(ContBlock, true);
    }
    // Emit final copy of the lastprivate variables if IsLastIter != 0.
    if (HasLastprivateClause) {
      CGF.EmitOMPLastprivateClauseFinal(
          S, isOpenMPSimdDirective(S.getDirectiveKind()),
          CGF.Builder.CreateIsNotNull(CGF.EmitLoadOfScalar(
              CGF.GetAddrOfLocalVar(*LIP), /*Volatile=*/false,
              (*LIP)->getType(), S.getLocStart())));
    }
  };
  auto &&TaskGen = [&S, SharedsTy, CapturedStruct,
                    IfCond](CodeGenFunction &CGF, llvm::Value *OutlinedFn,
                            const OMPTaskDataTy &Data) {
    auto &&CodeGen = [&](CodeGenFunction &CGF, PrePostActionTy &) {
      OMPLoopScope PreInitScope(CGF, S, /*EmitPreInit=*/true);
      CGF.CGM.getOpenMPRuntime().emitTaskLoopCall(CGF, S.getLocStart(), S,
                                                  OutlinedFn, SharedsTy,
                                                  CapturedStruct, IfCond, Data);
    };
    CGF.CGM.getOpenMPRuntime().emitInlinedDirective(CGF, OMPD_taskloop,
                                                    CodeGen);
  };
  if (Data.Nogroup)
    EmitOMPTaskBasedDirective(S, BodyGen, TaskGen, Data);
  else {
    CGM.getOpenMPRuntime().emitTaskgroupRegion(
        *this,
        [&S, &BodyGen, &TaskGen, &Data](CodeGenFunction &CGF,
                                        PrePostActionTy &Action) {
          Action.Enter(CGF);
          CGF.EmitOMPTaskBasedDirective(S, BodyGen, TaskGen, Data);
        },
        S.getLocStart());
  }
}

void CodeGenFunction::EmitOMPTaskLoopDirective(const OMPTaskLoopDirective &S) {
  EmitOMPTaskLoopBasedDirective(S);
}

void CodeGenFunction::EmitOMPTaskLoopSimdDirective(
    const OMPTaskLoopSimdDirective &S) {
  EmitOMPTaskLoopBasedDirective(S);
}

// Generate the instructions for '#pragma omp target update' directive.
void CodeGenFunction::EmitOMPTargetUpdateDirective(
    const OMPTargetUpdateDirective &S) {
  // If we don't have target devices, don't bother emitting the data mapping
  // code.
  if (CGM.getLangOpts().OMPTargetTriples.empty())
    return;

  // Check if we have any if clause associated with the directive.
  const Expr *IfCond = nullptr;
  if (auto *C = S.getSingleClause<OMPIfClause>())
    IfCond = C->getCondition();

  // Check if we have any device clause associated with the directive.
  const Expr *Device = nullptr;
  if (auto *C = S.getSingleClause<OMPDeviceClause>())
    Device = C->getDevice();

  CGM.getOpenMPRuntime().emitTargetDataStandAloneCall(*this, S, IfCond, Device);
}
