import { Component, OnInit } from '@angular/core';
import { CommonService } from '../common/common.service';

import * as three from 'three';
import { CubeService } from '../common/cube.service';

@Component({
  selector: 'app-rubiks',
  templateUrl: './rubiks.component.html',
  styleUrls: ["./rubiks.component.css"],
})
export class RubiksComponent {

  sides = ["F", "B", "T", "D", "L", "R"];
  indices = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  colours = ["red", "orange", "white", "yellow", "green", "blue"];

  constructor(public commonService: CommonService, public cubeService: CubeService) { }

  public getClass(i: number, j: number) {
    const colour = this.colours[this.cubeService.currentState69[i][j]];
    return `sticker ${colour}`;
  }

}
