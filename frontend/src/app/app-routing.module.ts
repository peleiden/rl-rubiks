import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { CubeComponent } from './cube/cube.component';
import { AppendixComponent } from './appendix/appendix.component';
import { DocumentationComponent } from './documentation/documentation.component';


const routes: Routes = [
  { path: "", redirectTo: "cube", pathMatch: "full" },
  { path: "cube", component: CubeComponent },
  { path: "appendix", component: AppendixComponent },
  { path: "documentation", component: DocumentationComponent },
  { path: "**", component: CubeComponent },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
