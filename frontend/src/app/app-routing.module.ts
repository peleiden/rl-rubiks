import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { CubeComponent } from './cube/cube.component';
// import { MethodsComponent } from './methods/methods.component';
import { DocumentationComponent } from './documentation/documentation.component';


const routes: Routes = [
  { path: "", redirectTo: "cube", pathMatch: "full" },
  { path: "cube", component: CubeComponent },
  // { path: "methods", component: MethodsComponent },
  { path: "documentation", component: DocumentationComponent },
  { path: "**", component: CubeComponent },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
