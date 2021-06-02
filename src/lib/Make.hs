{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}

module Make (loadDependencyGraph, upsweep, upsweepAndCodegen, TcGlobalEnv) where

import Control.Monad (foldM, unless)
import Control.Monad.IO.Class (liftIO)
import qualified Data.Graph as G
import qualified Data.List as L
import qualified Data.Map as M
import qualified Data.Text.Lazy as T

import Cxx.Syntax
import Env
import Magnolia.Check
import Magnolia.Parser
import Magnolia.PPrint
import Magnolia.Syntax
import Magnolia.Util
import MgToCxx

type TcGlobalEnv = Env (MPackage PhCheck)
type CxxCodegenEnv = M.Map FullyQualifiedName (CxxModule, [(TcTypeDecl, CxxName)])

upsweepAndCodegen :: [G.SCC PackageHead] -> MgMonad TcGlobalEnv
upsweepAndCodegen = (fst <$>) . foldMAccumErrorsAndFail go (M.empty, M.empty)
  where
    -- TODO: keep going?
    go
      :: (TcGlobalEnv, CxxCodegenEnv)
      -> G.SCC PackageHead
      -> MgMonad (TcGlobalEnv, CxxCodegenEnv)
    go _ (G.CyclicSCC pkgHead) =
      let pCycle = T.intercalate ", " $ map (pshow . _packageHeadName) pkgHead
      in throwNonLocatedE CyclicErr pCycle

    go (globalEnv, inCxxModules) (G.AcyclicSCC pkgHead) =
      let pkgName = fromFullyQualifiedName (_packageHeadName pkgHead)
      in enter pkgName $ do
        Ann ann (MPackage name decls deps) <-
          parsePackage (_packageHeadPath pkgHead) (_packageHeadStr pkgHead)
        -- TODO: add local/external ann for nodes
        importedEnv <- foldM (loadDependency globalEnv) M.empty deps
        -- 1. Renamings BROKEN
        envWithRenamings <- upsweepNamedRenamings importedEnv $
          topSortTopLevelE name (getNamedRenamings decls)
        -- TODO: deal with renamings first, then modules, then satisfactions
        -- 2. Modules
        (tcModulesNOTFINISHED, outCxxModules) <- upsweepAndCodegenModules
            pkgName (envWithRenamings, inCxxModules) $
            topSortTopLevelE name (getModules decls)
        -- 3. Satisfactions
        -- TODO: ^
        -- TODO: deal with deps and other tld types
        let tcPackage = Ann ann (MPackage name tcModulesNOTFINISHED [])
        return (M.insert name tcPackage globalEnv, outCxxModules)


upsweep :: [G.SCC PackageHead] -> MgMonad TcGlobalEnv
upsweep = foldMAccumErrorsAndFail go M.empty
  where
    -- TODO: keep going?
    go :: TcGlobalEnv
       -> G.SCC PackageHead
       -> MgMonad TcGlobalEnv
    go _ (G.CyclicSCC pkgHeads) =
      let pCycle = T.intercalate ", " $ map (pshow . _packageHeadName) pkgHeads
      in throwNonLocatedE CyclicErr pCycle

    go env (G.AcyclicSCC pkgHead) =
      enter (fromFullyQualifiedName (_packageHeadName pkgHead)) $ do
        pkg <- parsePackage (_packageHeadPath pkgHead) (_packageHeadStr pkgHead)
        tcPkg <- checkPackage env pkg
        return $ M.insert (nodeName pkg) tcPkg env

upsweepAndCodegenModules
  :: Name
  -> (Env [TcTopLevelDecl], CxxCodegenEnv)
  -> [G.SCC (MModule PhParse)]
  -> MgMonad (Env [TcTopLevelDecl], CxxCodegenEnv)
upsweepAndCodegenModules pkgName = foldMAccumErrors go
  where
    go
      :: (Env [TcTopLevelDecl], CxxCodegenEnv)
      -> G.SCC (MModule PhParse)
      -> MgMonad (Env [TcTopLevelDecl], CxxCodegenEnv)
    go _ (G.CyclicSCC modules) =
      let mCycle = T.intercalate ", " $ map (pshow . nodeName) modules in
      throwNonLocatedE CyclicErr mCycle

    -- TODO: error catching & recovery
    go (env, cxxModules) (G.AcyclicSCC modul) = do
      tcModule <- checkModule env modul
      let fqModuleName = FullyQualifiedName (Just pkgName) (nodeName modul)
      moduleCxx <- mgToCxx (fqModuleName, tcModule) cxxModules
      return ( M.insertWith (<>) (nodeName modul) [MModuleDecl tcModule] env
             , M.insert fqModuleName moduleCxx cxxModules)

-- TODO: cache and choose what to reload with granularity
-- | Takes the path towards a Magnolia package and constructs its dependency
-- graph, which is returned as a topologically sorted list of strongly
-- connected components (of package heads).
--
-- For example, assuming a Magnolia package "A" depends on a Magnolia package
-- "B", the result will be sorted like ["B", "A"].
loadDependencyGraph :: FilePath -> MgMonad [G.SCC PackageHead]
loadDependencyGraph = (topSortPackages <$>) . recover (go M.empty)
  where
    go :: M.Map Name PackageHead -> FilePath -> MgMonad (M.Map Name PackageHead)
    go loadedHeads filePath = do
      unless (".mg" `L.isSuffixOf` filePath) $
        throwNonLocatedE MiscErr
          "Magnolia source code files must have the \".mg\" extension"
      let expectedPkgName = mkPkgNameFromPath filePath
      case M.lookup expectedPkgName loadedHeads of
        Just _ -> return loadedHeads
        Nothing -> do
          input <- liftIO $ readFile filePath
          packageHead <- parsePackageHead filePath input
          let pkgName = fromFullyQualifiedName $ _packageHeadName packageHead
              imports =  map (mkPkgPathFromName . fromFullyQualifiedName)
                             (_packageHeadImports packageHead)
          unless (expectedPkgName == pkgName) $
            throwNonLocatedE MiscErr $ "expected package to have " <>
              "name " <> pshow expectedPkgName <> " but got " <>
              pshow pkgName
          foldM go (M.insert pkgName packageHead loadedHeads) imports
    topSortPackages pkgHeads = G.stronglyConnComp
        [ ( pkgHead
          , _packageHeadName pkgHead
          , _packageHeadImports pkgHead
          )
        | (_, pkgHead) <- M.toList pkgHeads
        ]