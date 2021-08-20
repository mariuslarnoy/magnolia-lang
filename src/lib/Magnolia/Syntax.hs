{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Magnolia.Syntax (
  -- * AST nodes
  --
  -- $mgAst
  -- TODO(bchetioui, 07/07/21): cleanup node names. This is put on hold to
  -- avoid too much rebasing of a big PR.
  -- ** Generic AST nodes
    CBody (..)
  , CGuard
  , InlineRenaming
  , MaybeTypedVar
  , MaybeTypedVar'
  , MBlockType (..)
  , MCallableDecl
  , MCallableDecl' (..)
  , MCallableType (..)
  , MDecl (..)
  , MExpr
  , MExpr' (..)
  , MModule
  , MModule' (..)
  , MModuleExpr
  , MModuleExpr' (..)
  , MModuleDep
  , MModuleDep' (..)
  , MModuleType (..)
  , MNamedRenaming
  , MNamedRenaming' (..)
  , MPackage
  , MPackage' (..)
  , MPackageDep
  , MPackageDep' (..)
  , MRenaming
  , MRenaming' (..)
  , MRenamingBlock
  , MRenamingBlock' (..)
  , MSatisfaction
  , MSatisfaction' (..)
  , MTopLevelDecl (..)
  , MType
  , MTypeDecl
  , MTypeDecl' (..)
  , MVar (..)
  , MVarMode (..)
  , TypedVar
  , TypedVar'
  -- ** AST nodes after the parsing phase
  , Parsed
  , ParsedCallableDecl
  , ParsedDecl
  , ParsedExpr
  , ParsedMaybeTypedVar
  , ParsedModule
  , ParsedModuleDep
  , ParsedModuleExpr
  , ParsedNamedRenaming
  , ParsedPackage
  , ParsedRenaming
  , ParsedRenamingBlock
  , ParsedSatisfaction
  , ParsedTopLevelDecl
  , ParsedTypeDecl
  , ParsedTypedVar
  -- ** AST nodes after the type checking phase
  , Tc
  , TcCallableDecl
  , TcDecl
  , TcExpr
  , TcMaybeTypedVar
  , TcModule
  , TcModuleDep
  , TcModuleExpr
  , TcNamedRenaming
  , TcPackage
  , TcRenaming
  , TcRenamingBlock
  , TcSatisfaction
  , TcTopLevelDecl
  , TcTypeDecl
  , TcTypedVar
  -- ** \"Primitive\" Magnolia types
  , pattern Pred
  , pattern Unit
  -- ** Utils
  , getCallableDecls
  , getModules
  , getNamedRenamings
  , getTypeDecls
  , moduleDecls
  -- * Classes
  , HasDependencies (..)
  , HasName (..)
  , HasSrcCtx (..)
  -- * Annotation utils
  -- ** Annotation types
  , AbstractDeclOrigin
  , ConcreteDeclOrigin
  , mkConcreteLocalDecl
  , DeclOrigin (..)
  , SrcCtx (..)
  -- ** Annotation-related patterns
  , pattern AbstractLocalDecl
  , pattern ConcreteExternalDecl
  , pattern ConcreteImportedMagnoliaDecl
  , pattern ConcreteLocalMagnoliaDecl
  , pattern ConcreteMagnoliaDecl
  -- ** Annotation wrapper utils
  , Ann (..)
  , XAnn
  , (<$$>)
  , (<$$)
  -- * Parsing utils
  , PackageHead (..)
  -- * Compilation phases
  , PhParse
  , PhCheck
  , PhCodeGen
  -- * Errors
  , Err (..)
  , ErrType (..)
  -- * Codegen
  , Backend (..)
  -- * Repl utils
  , Command (..)
  )
  where

import qualified Data.List.NonEmpty as NE
import qualified Data.Map as M
import qualified Data.Text.Lazy as T
import Data.Void

import Env

-- === repl utils ===

data Command = LoadPackage String
             | ReloadPackage String
             | InspectModule Name -- TODO: add optional package name
             | InspectPackage Name
             | ListPackages
             | ShowMenu

-- === preprocessing utils ===

data PackageHead = PackageHead { _packageHeadPath :: FilePath
                               , _packageHeadFileContent :: String
                               , _packageHeadName :: FullyQualifiedName
                               , _packageHeadImports :: [FullyQualifiedName]
                               , _packageHeadSrcCtx :: SrcCtx
                               }
                   deriving (Eq, Show)

-- === useful type aliases ===

type Tc e = Ann PhCheck e
type TcPackage = MPackage PhCheck
type TcTopLevelDecl = MTopLevelDecl PhCheck
type TcNamedRenaming = MNamedRenaming PhCheck
type TcRenamingBlock = MRenamingBlock PhCheck
type TcRenaming = MRenaming PhCheck
type TcModule = MModule PhCheck
type TcModuleDep = MModuleDep PhCheck
type TcModuleExpr = MModuleExpr PhCheck
type TcDecl = MDecl PhCheck
type TcCallableDecl = MCallableDecl PhCheck
type TcTypeDecl = MTypeDecl PhCheck
type TcExpr = MExpr PhCheck
type TcTypedVar = TypedVar PhCheck
type TcMaybeTypedVar = MaybeTypedVar PhCheck
type TcSatisfaction = MSatisfaction PhCheck

type Parsed e = Ann PhParse e
type ParsedPackage = MPackage PhParse
type ParsedTopLevelDecl = MTopLevelDecl PhParse
type ParsedNamedRenaming = MNamedRenaming PhParse
type ParsedRenamingBlock = MRenamingBlock PhParse
type ParsedRenaming = MRenaming PhParse
type ParsedModule = MModule PhParse
type ParsedModuleDep = MModuleDep PhParse
type ParsedModuleExpr = MModuleExpr PhParse
type ParsedDecl = MDecl PhParse
type ParsedCallableDecl = MCallableDecl PhParse
type ParsedTypeDecl = MTypeDecl PhParse
type ParsedExpr = MExpr PhParse
type ParsedTypedVar = TypedVar PhParse
type ParsedMaybeTypedVar = MaybeTypedVar PhParse
type ParsedSatisfaction = MSatisfaction PhParse

-- Ann [compilation phase] [node type]
data Ann p e = Ann { _ann :: XAnn p e
                   , _elem :: e p
                   }

instance Eq (e p) => Eq (Ann p e) where
  Ann _ e1 == Ann _ e2 = e1 == e2

instance Ord (e p) => Ord (Ann p e) where
  Ann _ e1 `compare` Ann _ e2 = e1 `compare` e2

-- TODO: display annotation using UndecidableInstances?
instance Show (e p) => Show (Ann p e) where
  show = show . _elem

(<$$>) :: XAnn p e ~ XAnn p' e' => (e p -> e' p') -> Ann p e -> Ann p' e'
(<$$>) f e = Ann { _ann = _ann e, _elem = f (_elem e) }

(<$$) :: XAnn p e ~ XAnn p' e' => e' p' -> Ann p e -> Ann p' e'
(<$$) e (Ann ann _) = Ann { _ann = ann, _elem = e }

-- === AST ===

-- $mgAst
--
-- The Magnolia AST defined here follows the patterns described in the
-- [Trees that Grow](https://www.microsoft.com/en-us/research/uploads/prod/2016/11/trees-that-grow.pdf)
-- paper to carry different types of annotations depending on the current
-- compiler phase.
--
-- When two types named T and T\' are defined, T is an annotated version of
-- T\', i.e. it is a parameterized type synonym of the form
-- > type T p = Ann p T'

-- TODO: use fully qualified names consistently at the package level
type MPackage p = Ann p MPackage'
data MPackage' p = MPackage { _packageName :: Name
                            , _packageDecls :: XPhasedContainer p (MTopLevelDecl p)
                            , _packageDeps :: [MPackageDep p]
                            }

type MPackageDep p = Ann p MPackageDep'
newtype MPackageDep' p = MPackageDep FullyQualifiedName
                         deriving (Eq, Show)

-- TODO: split in package?
data MTopLevelDecl p = MNamedRenamingDecl (MNamedRenaming p)
                     | MModuleDecl (MModule p)
                     | MSatisfactionDecl (MSatisfaction p)

type MNamedRenaming p = Ann p MNamedRenaming'
data MNamedRenaming' p = MNamedRenaming Name (MRenamingBlock p)

type MSatisfaction p = Ann p MSatisfaction'
data MSatisfaction' p =
  MSatisfaction Name (MModuleExpr p) (Maybe (MModuleExpr p)) (MModuleExpr p)

type MModule p = Ann p MModule'
data MModule' p = MModule MModuleType Name (MModuleExpr p)

type MModuleExpr p = Ann p MModuleExpr'
data MModuleExpr' p =
    MModuleDef (XPhasedContainer p (MDecl p)) [MModuleDep p]
               [MRenamingBlock p]
  | MModuleRef (XRef p) [MRenamingBlock p]

-- Expose out for DAG building
type MModuleDep p = Ann p MModuleDep'
-- Represents a dependency to a module with an associated list of renaming
-- blocks, as well as whether to only extract the signature of the dependency.
data MModuleDep' p = MModuleDep { _depName :: FullyQualifiedName
                                , _depRenamingBlocks :: [MRenamingBlock p]
                                , _depCastToSig :: Bool
                                }

type MRenamingBlock p = Ann p MRenamingBlock'
newtype MRenamingBlock' p = MRenamingBlock [MRenaming p]

type MRenaming p = Ann p MRenaming'
data MRenaming' p = InlineRenaming InlineRenaming
                  | RefRenaming (XRef p)

type InlineRenaming = (Name, Name)

data MModuleType = Signature
                 | Concept
                 | Implementation
                 | Program
                 | External Backend FullyQualifiedName
                   deriving (Eq, Show)

data MDecl p = MTypeDecl (MTypeDecl p)
             | MCallableDecl (MCallableDecl p)
               deriving (Eq, Show)

type MTypeDecl p = Ann p MTypeDecl'
data MTypeDecl' p = Type { _typeName :: MType, _typeIsRequired :: Bool }
                   deriving (Eq, Ord, Show)

type MCallableDecl p = Ann p MCallableDecl'
data MCallableDecl' p =
  Callable { _callableType :: MCallableType
           , _callableName :: Name
           , _callableArgs :: [TypedVar p]
           , _callableReturnType :: MType
           , _callableGuard :: CGuard p
           , _callableBody :: CBody p
           }
  deriving (Eq, Show)

-- TODO: at the moment, with only C++ as a backend, we assume any external body
--       comes from C++. When we actually implement other backends, we will need
--       to carry information about the external bodies. For instance, a file
--       can contain both a JS and a C++ implementation for the same external
--       functions. These two concrete implementations will be joinable, since
--       they are backend-dependent (and there is always only one backend).
--       This will need to be handled at the ConcreteDecl level as well.
data CBody p = ExternalBody | EmptyBody | MagnoliaBody (MExpr p)
               deriving (Eq, Show)
type CGuard p = Maybe (MExpr p)

type MType = Name

-- | Predicate type, used for conditionals.
pattern Pred :: MType
pattern Pred = GenName "Predicate"

-- | Unit type, used for stateful computations.
pattern Unit :: MType
pattern Unit = GenName "Unit"

data MCallableType = Axiom | Function | Predicate | Procedure
                    deriving (Eq, Show)

-- TODO: make a constructor for coercedexpr to remove (Maybe MType) from calls?
type MExpr p = Ann p MExpr'
data MExpr' p = MVar (MaybeTypedVar p)
              -- TODO: add Procedure/FunctionLike namespaces to Name?
              | MCall Name [MExpr p] (Maybe MType)
              | MBlockExpr MBlockType (NE.NonEmpty (MExpr p))
              | MValue (MExpr p)
              | MLet (MaybeTypedVar p) (Maybe (MExpr p))
              | MIf (MExpr p) (MExpr p) (MExpr p)
              | MAssert (MExpr p)
              | MSkip
                deriving (Eq, Show)

data MBlockType = MValueBlock | MEffectfulBlock
                  deriving (Eq, Show)

type TypedVar p = Ann p TypedVar'
type TypedVar' = MVar MType

type MaybeTypedVar p = Ann p MaybeTypedVar'
type MaybeTypedVar' = MVar (Maybe MType)

data MVar typAnnType p = Var { _varMode :: MVarMode
                             , _varName :: Name
                             , _varType :: typAnnType
                             }
                         deriving (Eq, Show)

-- Mode is either Obs (const), Out (unset ref), Upd (ref), or Unk(nown)
data MVarMode = MObs | MOut | MUnk | MUpd
                deriving (Eq, Show)

-- == code generation utils ==

data Backend = Cxx | JavaScript | Python
               deriving (Eq, Ord, Show)

-- == annotation utils ==

type SrcPos = (String, Int, Int)

newtype SrcCtx = SrcCtx (Maybe (SrcPos, SrcPos))
                 deriving (Eq, Ord, Show)

-- | An error generated by the compiler.
data Err = Err { -- | The type of the error.
                 _errorType :: ErrType
                 -- | The location associated with the error within the source
                 -- files.
               , _errorLoc :: SrcCtx
                 -- | The stack of scopes within which the error was thrown.
               , _errorParentScopes :: [Name]
                 -- | The text of the error.
               , _errorText :: T.Text
               }
           deriving (Eq, Show)

instance Ord Err where
  Err e1 src1 loc1 txt1 `compare` Err e2 src2 loc2 txt2 =
    (src1 `compare` src2) <>
    (loc1 `compare` loc2) <>
    (txt1 `compare` txt2) <>
    (e1 `compare` e2)

data ErrType = AmbiguousFunctionRefErr
             | AmbiguousProcedureRefErr
             | AmbiguousTopLevelRefErr
             | CompilerErr
             | CyclicErr
             | DeclContextErr
             | InvalidDeclErr
             | MiscErr
             | ModeMismatchErr
             | NotImplementedErr
             | ParseErr
             | TypeErr
             | UnboundFunctionErr
             | UnboundNameErr
             | UnboundProcedureErr
             | UnboundTopLevelErr
             | UnboundTypeErr
             | UnboundVarErr
               deriving (Eq, Ord, Show)

-- | Wraps the source location information of a declaration.
data DeclOrigin
  -- | Annotates a local declaration. At the module level, any declaration
  -- carries a local annotation. At the package level, only the modules
  -- defined in the current module should carry this annotation.
  = LocalDecl SrcCtx
  -- | Annotates an imported declaration. At the module level, no declaration
  -- carries such an annotation at the moment (TODO: should we change that
  -- behavior?). The reason is that within a module, declarations always have
  -- a corresponding local declaration. At the package level, all the modules
  -- imported from different packages should carry this annotation.
  | ImportedDecl FullyQualifiedName SrcCtx
    deriving (Eq, Show)

instance Ord DeclOrigin where
  compare declO1 declO2 = compare (srcCtx declO1) (srcCtx declO2)

-- | Wraps the source location information of a concrete declaration (i.e. a
-- declaration that is not required). If the inner constructor is Left, then
-- the origin of the declaration is interpreted as being internal (i.e. the
-- declaration is defined purely using Magnolia code). A contrario, if the
-- inner constructor is Right, then the origin of the declaration is interpreted
-- as being external (i.e. the declaration is defined to exist on some
-- backend). In that case, the declaration is also accompanied with the fully
-- qualified name that corresponds to the function in the external context,
-- and the relevant backend.
newtype ConcreteDeclOrigin =
  ConcreteDeclOrigin (Either DeclOrigin
                             (DeclOrigin, Backend, FullyQualifiedName))
  deriving (Eq, Show)

instance Ord ConcreteDeclOrigin where
  compare conDeclO1 conDeclO2 = compare (srcCtx conDeclO1) (srcCtx conDeclO2)

-- | Exposed constructor for ConcreteDeclOrigins. If a backend and a scope
-- name are provided, produces an external ConcreteDeclOrigin. Otherwise,
-- produces an internal one.
mkConcreteLocalDecl :: Maybe (Backend, Name) -> SrcCtx -> Name
                    -> ConcreteDeclOrigin
mkConcreteLocalDecl mexternalInfo src name = case mexternalInfo of
  Nothing -> ConcreteLocalMagnoliaDecl src
  Just (backend, scopeName) -> ConcreteLocalExternalDecl backend
    (FullyQualifiedName (Just scopeName) name) src

-- | Wraps the source location information of an abstract declaration (i.e. a
-- declaration that is required).
newtype AbstractDeclOrigin = AbstractDeclOrigin DeclOrigin
                             deriving (Eq, Ord)

-- === compilation phases ===

data PhParse
data PhCheck
data PhCodeGen

-- === XAnn type family ===

type family XAnn p (e :: * -> *) where
  XAnn PhParse MPackage' = SrcCtx
  XAnn PhCheck MPackage' = SrcCtx

  XAnn PhParse MPackageDep' = SrcCtx
  XAnn PhCheck MPackageDep' = SrcCtx

  XAnn PhParse MNamedRenaming' = SrcCtx
  XAnn PhCheck MNamedRenaming' = DeclOrigin

  XAnn PhParse MSatisfaction' = SrcCtx
  XAnn PhCheck MSatisfaction' = DeclOrigin

  XAnn PhParse MModule' = SrcCtx
  XAnn PhCheck MModule' = DeclOrigin

  XAnn PhParse MModuleExpr' = SrcCtx
  XAnn PhCheck MModuleExpr' = SrcCtx

  XAnn PhParse MModuleDep' = SrcCtx
  XAnn PhCheck MModuleDep' = SrcCtx

  XAnn PhParse MRenamingBlock' = SrcCtx
  XAnn PhCheck MRenamingBlock' = SrcCtx

  XAnn PhParse MRenaming' = SrcCtx
  XAnn PhCheck MRenaming' = DeclOrigin

  XAnn PhParse MTypeDecl' = SrcCtx
  XAnn PhCheck MTypeDecl' = ( Maybe ConcreteDeclOrigin
                           , NE.NonEmpty AbstractDeclOrigin
                           )

  XAnn PhParse MCallableDecl' = SrcCtx
  XAnn PhCheck MCallableDecl' = ( Maybe ConcreteDeclOrigin
                               , NE.NonEmpty AbstractDeclOrigin
                               )

  XAnn PhParse MExpr' = SrcCtx
  XAnn PhCheck MExpr' = MType

  XAnn PhParse (MVar _) = SrcCtx
  XAnn PhCheck (MVar _) = MType

-- === other useful type families ===

type family XPhasedContainer p e where
  XPhasedContainer PhParse e = [e]
  XPhasedContainer PhCheck e = M.Map Name [e]

-- | The goal of XRef is to statically prevent the existence of references to
-- named top level elements after the consistency/type checking phase.
type family XRef p where
  XRef PhParse = FullyQualifiedName
  XRef PhCheck = Void

-- === standalone show instances ===

deriving instance Show (MRenaming' PhCheck)
deriving instance Show (MRenamingBlock' PhCheck)
deriving instance Show (MModuleDep' PhCheck)
deriving instance Show (MNamedRenaming' PhCheck)
deriving instance Show (MModule' PhCheck)
deriving instance Show (MModuleExpr' PhCheck)
deriving instance Show (MSatisfaction' PhCheck)
deriving instance Show (MTopLevelDecl PhCheck)
deriving instance Show (MPackage' PhCheck)

-- === useful typeclasses ===

class HasName n where
  nodeName :: n -> Name

instance HasName PackageHead where
  nodeName = fromFullyQualifiedName  . _packageHeadName

instance HasName (e p) => HasName (Ann p e) where
  nodeName = nodeName . _elem

instance HasName (MPackage' p) where
  nodeName = _packageName

instance HasName (MPackageDep' p) where
  nodeName (MPackageDep name) = fromFullyQualifiedName name

instance HasName (MTopLevelDecl p) where
  nodeName topLevelDecl = case topLevelDecl of
    MNamedRenamingDecl namedRenaming -> nodeName namedRenaming
    MModuleDecl modul -> nodeName modul
    MSatisfactionDecl satisfaction -> nodeName satisfaction

instance HasName (MNamedRenaming' p) where
  nodeName (MNamedRenaming name _) = name

instance HasName (MModule' p) where
  nodeName (MModule _ name _) = name

instance HasName (MSatisfaction' p) where
  nodeName (MSatisfaction name _ _ _) = name

instance HasName (MModuleDep' p) where
  nodeName (MModuleDep name _ _) = fromFullyQualifiedName name

instance HasName (MDecl p) where
  nodeName decl = case decl of
    MTypeDecl tdecl -> nodeName tdecl
    MCallableDecl cdecl -> nodeName cdecl

instance HasName (MTypeDecl' p) where
  nodeName (Type name _) = name

instance HasName (MCallableDecl' p) where
  nodeName (Callable _ name _ _ _ _) = name

instance HasName (MVar typAnnType p) where
  nodeName (Var _ name _) = name

class HasDependencies a where
  dependencies :: a -> [FullyQualifiedName]

instance HasDependencies (e p) => HasDependencies (Ann p e) where
  dependencies = dependencies . _elem

instance HasDependencies PackageHead where
  dependencies = _packageHeadImports

instance HasDependencies (MModule' PhParse) where
  dependencies (MModule _ _ moduleExpr) = dependencies moduleExpr

instance HasDependencies (MModule' PhCheck) where
  dependencies (MModule _ _ moduleExpr) = dependencies moduleExpr

instance HasDependencies (MModuleExpr' PhParse) where
  dependencies moduleExpr = case moduleExpr of
    MModuleDef _ deps _ -> map (_depName . _elem) deps
    MModuleRef refName _ -> [refName]

instance HasDependencies (MModuleExpr' PhCheck) where
  dependencies modul = case modul of
    MModuleDef _ deps _ -> map (_depName . _elem) deps
    MModuleRef v _ -> absurd v

instance HasDependencies (MNamedRenaming' PhParse) where
  dependencies (MNamedRenaming _ renamingBlock) =
    dependencies renamingBlock

instance HasDependencies (MRenamingBlock' PhParse) where
  dependencies (MRenamingBlock renamings) =
    foldr (\(Ann _ r) acc -> case r of RefRenaming n -> n:acc ; _ -> acc) []
          renamings

instance HasDependencies (MPackage' p) where
  dependencies (MPackage _ _ deps) =
    map (\(Ann _ (MPackageDep depName)) -> depName) deps

class HasSrcCtx a where
  srcCtx :: a -> SrcCtx

instance HasSrcCtx SrcCtx where
  srcCtx = id

instance HasSrcCtx PackageHead where
  srcCtx = _packageHeadSrcCtx

instance HasSrcCtx DeclOrigin where
  srcCtx declO = case declO of
    LocalDecl src -> src
    ImportedDecl _ src -> src

instance HasSrcCtx ConcreteDeclOrigin where
  srcCtx (ConcreteDeclOrigin edeclO) = case edeclO of
    Left declO -> srcCtx declO
    Right (declO, _, _) -> srcCtx declO

instance HasSrcCtx AbstractDeclOrigin where
  srcCtx (AbstractDeclOrigin declO) = srcCtx declO

instance HasSrcCtx (SrcCtx, a) where
  srcCtx (src, _) = src

instance HasSrcCtx (XAnn p e) => HasSrcCtx (Ann p e) where
  srcCtx = srcCtx . _ann

-- === useful patterns ===

pattern AbstractLocalDecl :: SrcCtx -> AbstractDeclOrigin
pattern AbstractLocalDecl src = AbstractDeclOrigin (LocalDecl src)

pattern ConcreteExternalDecl :: DeclOrigin -> Backend -> FullyQualifiedName
                             -> ConcreteDeclOrigin
pattern ConcreteExternalDecl declO backend fqn =
  ConcreteDeclOrigin (Right (declO, backend, fqn))

pattern ConcreteMagnoliaDecl :: DeclOrigin
                             -> ConcreteDeclOrigin
pattern ConcreteMagnoliaDecl declO = ConcreteDeclOrigin (Left declO)

{-# COMPLETE ConcreteExternalDecl, ConcreteMagnoliaDecl #-}

pattern ConcreteImportedMagnoliaDecl
  :: FullyQualifiedName -> SrcCtx -> ConcreteDeclOrigin
pattern ConcreteImportedMagnoliaDecl fqn src =
  ConcreteDeclOrigin (Left (ImportedDecl fqn src))

pattern ConcreteLocalMagnoliaDecl :: SrcCtx -> ConcreteDeclOrigin
pattern ConcreteLocalMagnoliaDecl src =
  ConcreteDeclOrigin (Left (LocalDecl src))

pattern ConcreteLocalExternalDecl :: Backend -> FullyQualifiedName -> SrcCtx
                                  -> ConcreteDeclOrigin
pattern ConcreteLocalExternalDecl backend fqn src =
  ConcreteDeclOrigin (Right (LocalDecl src, backend, fqn))

-- === top level declarations manipulation ===

getModules :: Foldable t => t (MTopLevelDecl p) -> [MModule p]
getModules = foldl extractModule []
  where
    extractModule :: [MModule p] -> MTopLevelDecl p -> [MModule p]
    extractModule acc topLevelDecl
      | MModuleDecl m <- topLevelDecl = m:acc
      | otherwise = acc

getNamedRenamings :: Foldable t => t (MTopLevelDecl p) -> [MNamedRenaming p]
getNamedRenamings = foldl extractNamedRenaming []
  where
    extractNamedRenaming
      :: [MNamedRenaming p] -> MTopLevelDecl p -> [MNamedRenaming p]
    extractNamedRenaming acc topLevelDecl
      | MNamedRenamingDecl nr <- topLevelDecl = nr:acc
      | otherwise = acc

-- === modules manipulation ===

moduleDecls :: MModule PhCheck -> Env [TcDecl]
moduleDecls (Ann _ (MModule _ _ (Ann _ moduleExpr))) = case moduleExpr of
  MModuleDef decls _ _ -> decls
  MModuleRef v _ -> absurd v

-- === module declarations manipulation ===

getTypeDecls :: Foldable t => t (MDecl p) -> [MTypeDecl p]
getTypeDecls = foldr extractType []
  where
    extractType :: MDecl p -> [MTypeDecl p] -> [MTypeDecl p]
    extractType decl acc = case decl of
      MTypeDecl tdecl -> tdecl:acc
      _ -> acc

getCallableDecls :: Foldable t => t (MDecl p) -> [MCallableDecl p]
getCallableDecls = foldr extractCallable []
  where
    extractCallable :: MDecl p -> [MCallableDecl p] -> [MCallableDecl p]
    extractCallable decl acc = case decl of
      MCallableDecl cdecl -> cdecl:acc
      _ -> acc