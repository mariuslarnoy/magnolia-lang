{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}

module Util (
  -- * MgMonad utils
    MgMonad
  , runMgMonad
  -- ** MgMonad error-related utils
  , enter
  , foldMAccumErrors
  , foldMAccumErrorsAndFail
  , recover
  , recoverWithDefault
  , throwLocatedE
  , throwNonLocatedE
  -- * Package name manipulation utils
  , mkPkgNameFromPath
  , mkPkgPathFromName
  , mkPkgPathFromStr
  , isPkgPath
  -- * Top-level manipulation utils
  , lookupTopLevelRef
  -- * Renaming manipulation utils
  , mkInlineRenamings
  , expandRenaming
  , expandRenamingBlock
  , checkRenamingBlock
  -- * Declaration manipulation utils
  , isLocalDecl
  -- * Expression manipulations utils
  , isValueExpr
  )
  where

import Control.Monad
import Control.Monad.IO.Class (MonadIO (..))
import Control.Monad.Trans.Class (lift)
import qualified Control.Monad.Trans.Except as E
import qualified Control.Monad.Trans.Reader as R
import qualified Control.Monad.Trans.State as St
import qualified Data.List as L
import qualified Data.Map as M
import Data.Maybe (fromJust, isNothing)
import qualified Data.Set as S
import Data.Text.Prettyprint.Doc (Pretty)
import qualified Data.Text.Lazy as T
import Data.Void

import Env
import PPrint
import Syntax

-- === magnolia monad utils ===

-- | A monad transformer wrapping the minimum stack of monads required within
-- the compiler.
newtype MgMonadT e r s m a =
  MgMonadT { unMgT :: E.ExceptT e (R.ReaderT r (St.StateT s m)) a }
  deriving (Functor, Applicative, Monad)

instance MonadIO m => MonadIO (MgMonadT e r s m) where
  liftIO = MgMonadT . liftIO

type MgMonad = MgMonadT () [Name] (S.Set Err) IO

runMgMonadT :: MgMonadT e r s m a -> r -> s -> m (Either e a, s)
runMgMonadT mgm r s =
  (`St.runStateT` s) . (`R.runReaderT` r) . E.runExceptT $ unMgT mgm

runMgMonad :: MgMonad a -> IO (Either () a, S.Set Err)
runMgMonad m = runMgMonadT m [] S.empty

-- | Retrieves the state within a MgMonadT.
get :: Monad m => MgMonadT e r s m s
get = MgMonadT (lift (lift St.get))

-- | Modifies the state within a MgMonadT using the provided function.
modify :: Monad m => (s -> s) -> MgMonadT e r s m ()
modify f = MgMonadT (lift (lift (St.modify f)))

-- | Throws an exception within a MgMonadT.
throwE :: Monad m => e -> MgMonadT e s r m a
throwE = MgMonadT . E.throwE

-- | Catches an exception within a MgMonadT.
catchE :: Monad m => MgMonadT e r s m a -> (e -> MgMonadT e r s m a)
       -> MgMonadT e r s m a
catchE me f = MgMonadT (unMgT me `E.catchE` (unMgT . f))

-- | Retrieves the environment within a MgMonadT.
ask :: Monad m => MgMonadT e r s m r
ask = MgMonadT (lift R.ask)

-- | Executes a computation in a modified environment within a MgMonadT.
local :: Monad m => (r -> r) -> MgMonadT e r s m a -> MgMonadT e r s m a
local f = MgMonadT . E.mapExceptT (R.local f) . unMgT

-- === error handling utils ===

-- | Accumulates exceptions for each element in the traversable when performing
-- a fold. Always succeeds.
foldMAccumErrors :: Foldable t => (b -> a -> MgMonad b) -> b -> t a -> MgMonad b
foldMAccumErrors f = foldM (\b a -> f b a `catchE` const (return b))

-- | Similar to foldMAccumErrors, except that if any exception is caught
-- during processing, an exception is thrown at the end of the fold.
foldMAccumErrorsAndFail
  :: Foldable t => (b -> a -> MgMonad b) -> b -> t a -> MgMonad b
foldMAccumErrorsAndFail f initialValue elems = do
    foldMRes <- foldMAccumErrors f initialValue elems
    errs <- get
    maybe (return foldMRes) (const (throwE ())) (S.lookupMin errs)

-- | Runs a computation and recovers by returning a (generated) default value
-- if an exception is caught.
recover :: Monoid b => (a -> MgMonad b) -> (a -> MgMonad b)
recover f a = f a `catchE` const (return mempty)

-- | Similar to recover, except that the default value to return is provided
-- by the caller.
recoverWithDefault :: (a -> MgMonad b) -> b -> (a -> MgMonad b)
recoverWithDefault f b a = f a `catchE` const (return b)

-- | Returns the list of the parent scopes associated with the computation.
-- The list is ordered from the outermost to the innermost scope.
parentScopes :: MgMonad [Name]
parentScopes = L.reverse <$> ask

-- | Runs a computation within a child scope.
enter :: Name      -- ^ the name of the child scope
      -> MgMonad a -- ^ the computation
      -> MgMonad a
enter = local . (:)

-- | Throws an error with source information.
throwLocatedE :: ErrType -- ^ the type of error that was encountered
              -> SrcCtx  -- ^ the location of the error within the source
              -> T.Text  -- ^ the error text
              -> MgMonad b
throwLocatedE errType src err = do
  ps <- parentScopes
  modify (S.insert (Err errType src ps err)) >> throwE ()

-- | Like throwLocatedE, except that the source information is omitted.
throwNonLocatedE :: ErrType -> T.Text -> MgMonad b
throwNonLocatedE errType = throwLocatedE errType (SrcCtx Nothing)

-- === package name manipulation ===

mkPkgNameFromPath :: String -> Name
mkPkgNameFromPath pkgPath = PkgName $ map (\c -> if c == '/' then '.' else c) $
  take (length pkgPath - 3) pkgPath

mkPkgPathFromName :: Name -> String
mkPkgPathFromName = mkPkgPathFromStr. _name

mkPkgPathFromStr :: String -> String
mkPkgPathFromStr pkgStr =
  map (\c -> if c == '.' then '/' else c) pkgStr <> ".mg"

isPkgPath :: String -> Bool
isPkgPath s = ".mg" `L.isSuffixOf` s

-- === top-level manipulation ===

lookupTopLevelRef
  :: ( XAnn PhCheck e ~ DeclOrigin
     , Show (e PhCheck)
     , Pretty (e PhCheck)
     , NamedNode (e PhCheck)
     )
  => SrcCtx
  -> Env [Ann PhCheck e]
  -> FullyQualifiedName
  -> MgMonad (Ann PhCheck e)
lookupTopLevelRef src env ref@(FullyQualifiedName mscopeName targetName) =
  case M.lookup targetName env of
    Nothing -> throwLocatedE UnboundTopLevelErr src $ pshow ref
    Just matches ->
      if null matches then throwLocatedE CompilerErr src $
        "name " <> pshow targetName <> " exists but is not bound to anything"
      else do
        compatibleMatches <-
            if isNothing mscopeName then do
              -- There are two valid cases in which it is possible not to
              -- provide a scope name:
              -- (1) there is only one element in scope corresponding to the
              --     target name, and the code is therefore unambiguous; in
              --     this case, the name of the scope is here optional;
              -- (2) there is exactly one locally defined element in scope.
              --     In this case, the name of the scope *can not* be provided,
              --     because the surrounding package is not yet *in scope*.
              let localDecls = filter isLocalDecl matches
              if null localDecls then return matches
              else do
                unless (null $ tail localDecls) $
                  throwLocatedE CompilerErr src $
                    "there are several locally defined top-level constructs " <>
                    "with the name " <> pshow targetName <> " in scope"
                return localDecls
            else return $ filter (matchesImportScope (fromJust mscopeName))
                                  matches
        -- The reference is only valid when a match exists and is unambiguous,
        -- i.e. in cases when compatibleMatches contains exactly one element.
        when (null compatibleMatches) $ -- no match
          throwLocatedE UnboundTopLevelErr src $ pshow ref
        unless (null $ tail compatibleMatches) $ -- more than one match
          throwLocatedE AmbiguousTopLevelRefErr src $ pshow ref <>
            "'. Candidates are: " <> pshow (map mkFQName compatibleMatches)
        return $ head compatibleMatches
  where
    matchesImportScope
      :: XAnn PhCheck e ~ DeclOrigin => Name -> Ann PhCheck e -> Bool
    matchesImportScope scopeName (Ann declO _) = case declO of
      ImportedDecl (FullyQualifiedName (Just scopeName') _) _ ->
        scopeName == scopeName'
      _ -> False

    mkFQName
      :: (XAnn PhCheck e ~ DeclOrigin, NamedNode (e PhCheck))
      => Ann PhCheck e -> Name
    mkFQName (Ann declO node) = case declO of
      LocalDecl {} -> nodeName node
      ImportedDecl fqn _ -> fromFullyQualifiedName fqn

-- === renamings manipulation ===

-- TODO: coercion w/o RefRenaming type?
checkRenamingBlock
  :: MRenamingBlock PhParse
  -> MgMonad (MRenamingBlock PhCheck)
checkRenamingBlock (Ann blockSrc (MRenamingBlock renamings)) = do
  checkedRenamings <- traverse checkInlineRenaming renamings
  return $ Ann blockSrc $ MRenamingBlock checkedRenamings
  where
    checkInlineRenaming
      :: MRenaming PhParse
      -> MgMonad (MRenaming PhCheck)
    checkInlineRenaming (Ann src renaming) = case renaming of
      -- It seems that the below pattern matching can not be avoided, due to
      -- coercion concerns (https://gitlab.haskell.org/ghc/ghc/-/issues/15683).
      InlineRenaming r -> return $
        Ann (LocalDecl src) (InlineRenaming r)
      RefRenaming _ -> throwLocatedE CompilerErr src $ "references left " <>
        "in renaming block at checking time"

expandRenamingBlock
  :: Env [MNamedRenaming PhCheck]
  -> MRenamingBlock PhParse
  -> MgMonad (MRenamingBlock PhCheck)
expandRenamingBlock env (Ann src (MRenamingBlock renamings)) =
  Ann src . MRenamingBlock <$>
    (foldl (<>) [] <$> mapM (expandRenaming env) renamings)

expandRenaming
  :: Env [MNamedRenaming PhCheck]
  -> MRenaming PhParse
  -> MgMonad [MRenaming PhCheck]
expandRenaming env (Ann src renaming) = case renaming of
  InlineRenaming ir -> return
    [Ann (LocalDecl src) (InlineRenaming ir)]
  RefRenaming ref -> do
    Ann _ (MNamedRenaming _ (Ann _ (MRenamingBlock renamings))) <-
      lookupTopLevelRef src env ref
    return renamings

mkInlineRenamings :: MRenamingBlock PhCheck -> [InlineRenaming]
mkInlineRenamings (Ann _ (MRenamingBlock renamings)) =
    map mkInlineRenaming renamings
  where
    mkInlineRenaming :: MRenaming PhCheck -> InlineRenaming
    mkInlineRenaming (Ann _ renaming) = case renaming of
      InlineRenaming ir -> ir
      RefRenaming v -> absurd v

-- === declaration manipulation ===

isLocalDecl :: XAnn p e ~ DeclOrigin => Ann p e -> Bool
isLocalDecl (Ann declO _) = case declO of LocalDecl {} -> True ; _ -> False

-- === expression manipulation ===

isValueExpr :: MExpr p -> Bool
isValueExpr (Ann _ e) = case e of
  MValue _ -> True
  MIf _ eTrue eFalse -> isValueExpr eTrue || isValueExpr eFalse
  _ -> False